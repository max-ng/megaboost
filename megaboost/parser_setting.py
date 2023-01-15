import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--dataset', dest='dataset',
                    help='Name of the dataset',
                    default='CUSTOM', type=str)
parser.add_argument('--mode', dest='mode',
                    help='image/text',
                    default='image', type=str)
parser.add_argument('--number-of-class', dest='number_of_class',
                    help='number of class',
                    default=2, type=int)
parser.add_argument('--device', default='gpu', type=str, help='device')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--warmup-steps', default=10, type=int, help='warmup steps')
parser.add_argument("--amp", default=True, action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--temperature', default=1.25, type=float, help='pseudo label temperature')
parser.add_argument('--threshold', default=0.6, type=float, help='pseudo label threshold')
parser.add_argument('--lambda-u', default=8, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=500, type=float, help='warmup steps of lambda-u')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument('--dense-dropout', default=0.5, type=float, help='dropout on last dense layer')
parser.add_argument('--dropout', default=0, type=float, help='dropout on layer')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')