from typing import Optional
import os

import torch
from torch.cuda import amp

from .model import initialize_model, ModelEMA
from .utils import get_loss_function, get_cosine_schedule_with_warmup
from .main import train_loop, train_basic, validate, predict
from .data import prepare_cifar10
# from .torch_custom_dataset import CustomDataset

def mps_is_available():
    if not torch.backends.mps.is_available():
        return False
    else:
        return True

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

class MegaBoost:
    def __init__(self, config: Optional[dict] = None):
        self.mode = "train"
        self.epochs = 300000
        self.lr = 0.05
        self.model = "wideResNet-28-2"
        self.classes = 10

        self.device = "gpu"
        self.momentum = 0.9
        self.nesterov = True
        self.label_smoothing = 0.15 
        self.warmup_steps = 50
        self.amp = True
        self.print_freq = 100
        self.temperature = 1.25
        self.threshold = 0.6
        self.dense_dropout = 0.5
        self.dropout = 0
        self.eval_step = 1000
        self.ema = 0.995
        self.weight_decay = 5e-4
        self.best_top1 = 0 
        self.save_dir = "result"
        self.start_epoch = 0
        self.lambda_u = 8
        self.uda_steps = round_to_multiple(int(self.epochs * 0.01666666), 10)
        self.grad_clip = 0
        self.save_every = 1
        self.gpu = False

        if config is not None:
            for item in config:
                setattr(self, item, config[item])

        self.model = initialize_model(self)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.mode == "train" and self.device != "gpu":
            print("Stop: currently semi-supervised mode supports gpu only.")
            return

        if self.classes > 1000 and self.model != "wideResNet-28-2":
            print("The current model does not support 1000+ classes")
            return

        if mps_is_available():
            self.device = torch.device("mps")
            self.amp = False
            print("Mps is available")
        elif self.device == "cpu":
            self.device = torch.device("cpu")
            self.amp = False
        elif self.device == "gpu":
            self.device = torch.device("cuda")
            self.gpu = True
            torch.backends.cudnn.benchmark = True

        self.model.to(self.device)

        self.avg_model = None
        if self.ema > 0:
            self.avg_model = ModelEMA(self.model, self.ema)
        self.criterion = get_loss_function(self)

        if self.mode == "train":
            self.first_optimizer = torch.optim.SGD(self.model.parameters(),
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay,
                            nesterov=self.nesterov)

            self.second_optimizer = torch.optim.SGD(self.model.parameters(),
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay,
                            nesterov=self.nesterov)
            
            self.first_scheduler = get_cosine_schedule_with_warmup(self.first_optimizer,
                                                  0,
                                                  self.epochs)
                                                  

            self.second_scheduler = get_cosine_schedule_with_warmup(self.second_optimizer,
                                                  self.warmup_steps,
                                                  self.epochs)

            self.first_scaler = amp.GradScaler(enabled=self.amp)
            self.second_scaler = amp.GradScaler(enabled=self.amp)

        elif self.mode == "fintune-basic":
            self.first_optimizer = torch.optim.SGD(self.model.parameters(),
                            lr=self.lr,
                            momentum=self.momentum,
                            weight_decay=self.weight_decay,
                            nesterov=self.nesterov)
            
            self.first_scheduler = get_cosine_schedule_with_warmup(self.first_optimizer,
                                                  0,
                                                  self.epochs)

            self.first_scaler = amp.GradScaler(enabled=self.amp)

        else:
            raise NotImplementedError("Other modes are not ready.")

    def fit(self, X, T, UX = None):
        if self.mode == "train":
            train_loop(self, X, UX, T, self.model, self.avg_model, self.criterion, self.first_optimizer, self.second_optimizer, self.first_scheduler, self.second_scheduler, self.first_scaler, self.second_scaler, self.best_top1)
        
        elif self.mode == "fintune-basic":
            train_basic(self, X, T, self.model, self.avg_model, self.criterion, self.first_optimizer, self.first_scheduler, self.first_scaler, self.best_top1)

        return

    def validate(self, X):
        test_model = self.avg_model if self.avg_model is not None else self.model
        validate(self, X, test_model, self.criterion)
        return

    def predict(self, input, transform = None):
        test_model = self.avg_model if self.avg_model is not None else self.model
        res = predict(self, input, test_model, transform=transform)
        return res