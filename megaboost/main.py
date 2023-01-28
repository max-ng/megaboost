import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda import amp

from .model import get_model, ModelEMA
from .data import get_cifar10, get_toy
from .parser_setting import parser
from .utils import get_loss_function, get_cosine_schedule_with_warmup, AverageMeter, accuracy


def initialize_components(args):
    labeled_loader, unlabeled_loader, test_loader = get_dataset_dataloader(args)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.dataset == "CIFAR10" :
        args.number_of_class = 10
    elif args.dataset == 'CIFAR100':
        args.number_of_class = 100
    elif args.dataset == 'TWOMOON':
        args.number_of_class = 2

    # model = torch.nn.DataParallel(get_model(args))
    model = get_model(args)
    if mps_is_available():
        args.device = torch.device("mps")
    elif args.device == "cpu":
        args.device = torch.device("cpu")
    elif args.device == "gpu":
        args.device = torch.device("cuda")
    model.to(args.device)
    # model = ModelEMA(model, 0.995)
    avg_model = None
    if args.ema > 0:
        avg_model = ModelEMA(model, args.ema)
    cudnn.benchmark = True
    criterion = get_loss_function(args)

    first_optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    second_optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    # first_optimizer = torch.optim.SGD( [{'params': model.parameters(), 'initial_lr': args.lr}] , args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)
    # second_optimizer = torch.optim.SGD( [{'params': model.parameters(), 'initial_lr': args.lr}] , args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)

    first_scheduler = get_cosine_schedule_with_warmup(first_optimizer,
                                                  0,
                                                  args.epochs)
                                                  

    second_scheduler = get_cosine_schedule_with_warmup(second_optimizer,
                                                  args.warmup_steps,
                                                  args.epochs)

    first_scaler = amp.GradScaler(enabled=args.amp)
    second_scaler = amp.GradScaler(enabled=args.amp)

    return model, avg_model, labeled_loader, unlabeled_loader, test_loader, criterion, first_optimizer, second_optimizer, first_scheduler, second_scheduler, first_scaler, second_scaler

def get_dataset_dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    train_transformation_list = []
    test_transformation_list = []
    if args.mode == 'image':
        train_transformation_list += [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                normalize]
        test_transformation_list += [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                normalize]

    train_transformations = transforms.Compose(train_transformation_list)
    test_transformations = transforms.Compose(test_transformation_list)
    if args.dataset == "CIFAR10" :
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args)

    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transformations, download=True)
        test_dataset = datasets.CIFAR100(root='./data', train=False, transform=test_transformations, download=True)

    elif args.dataset == 'TWOMOON':
        labeled_dataset, unlabeled_dataset, test_dataset = get_toy(args)

    labeled_loader = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=args.batch_size*args.mu, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return labeled_loader, unlabeled_loader, test_loader

def mps_is_available():
    if not torch.backends.mps.is_available():
        return False
    else:
        return True

def train_loop(args, labeled_loader, unlabeled_loader, test_loader, model, avg_model, criterion, first_optimizer, second_optimizer, first_scheduler, second_scheduler, first_scaler, second_scaler, best_top1):
    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    model.zero_grad()

    

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    
    start = time.time()
    end = time.time()

    for epoch in range(args.start_epoch, args.epochs):
            model.train()
            try:
                images_l, target = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_loader)
                images_l, target = next(labeled_iter)
            try:
                (images_uw, images_us), _ = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_loader)
                (images_uw, images_us), _ = next(unlabeled_iter)

            images_l = images_l.to(args.device)
            images_uw = images_uw.to(args.device)
            images_us = images_us.to(args.device)
            target = target.to(args.device)

            with amp.autocast(enabled=args.amp):
                batch_size = images_l.shape[0]
                t_images = torch.cat((images_l, images_uw, images_us))
                t_logits = model(t_images)
                t_logits_l = t_logits[:batch_size]
                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)

                t_loss_l = criterion(t_logits_l, target)

                soft_pseudo_label = torch.softmax(t_logits_uw.detach()/args.temperature, dim=-1)
                max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                t_loss_u = torch.mean(
                    -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
                )
                weight_u = args.lambda_u * min(1., (epoch+1) / args.uda_steps)
                t_loss_uda = t_loss_l + weight_u * t_loss_u
                # t_loss_uda = t_loss_l
                s_logits = t_logits
                s_logits_l = t_logits_l
                s_logits_us = t_logits_us

                del s_logits

                s_loss_l_old = F.cross_entropy(s_logits_l.detach(), target)
                s_loss = criterion(s_logits_us, hard_pseudo_label)

            if args.amp:
                first_scaler.scale(s_loss).backward(retain_graph=True)
                first_scaler.unscale_(first_optimizer)
            else:
                first_optimizer.zero_grad()
                second_optimizer.zero_grad()
                s_loss.requires_grad_(True)
                s_loss.backward(retain_graph=True)

            clone_lr = first_optimizer.param_groups[0]['lr'] 
            clone_grad = {}
            clone_data = {}

            if args.gpu:          
                if not sum (v for v in first_scaler._found_inf_per_device(first_optimizer).values() ):
                    for name, p in model.named_parameters():
                        if p.data is not None and p.grad is not None:
                            # p.grad = torch.clamp(p.grad, min=-.1, max=.1)
                            clone_grad[name] = p.grad.clone().detach()
                            clone_data[name] = p.data.clone()
                            p.data -= clone_lr * clone_grad[name]   
                            p.grad.data.zero_()
                else:
                    print('inf values')
            else:
                for name, p in model.named_parameters():
                    if p.data is not None and p.grad is not None:
                        # p.grad = torch.clamp(p.grad, min=-.1, max=.1)
                        clone_grad[name] = p.grad.clone().detach()
                        clone_data[name] = p.data.clone()
                        p.data -= clone_lr * clone_grad[name]   
                        p.grad.data.zero_()


            if args.amp:
                first_scaler.update()
            first_scheduler.step()

            if args.ema > 0:
                avg_model.update_parameters(model)

            with amp.autocast(enabled=args.amp):
                with torch.no_grad(): 
                    s_logits_l = model(images_l)
                s_loss_l_new = F.cross_entropy(s_logits_l.detach(), target)
                dot_product = s_loss_l_new - s_loss_l_old
                moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
                dot_product = dot_product - moving_dot_product
                _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
                t_loss_mpl = dot_product * F.cross_entropy(t_logits_us , hard_pseudo_label)
                t_loss = t_loss_uda + 1*t_loss_mpl

            if args.amp:
                second_scaler.scale(t_loss).backward()
            else:
                t_loss.requires_grad_(True)
                t_loss.backward()

            if args.grad_clip > 0:
                second_scaler.unscale_(second_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.amp:
                second_scaler.step(second_optimizer)
                second_scaler.update()
            else:
                second_optimizer.step()

            second_scheduler.step()

            model.zero_grad()

            output = s_logits_l.detach().float()
            loss = s_loss_l_new.float()
            # Measure accuracy and record loss
            top1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), target.size(0))
            top1s.update(top1.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                print('Epoch: [{0}]\t'
                      'Elapsed: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss: {loss.val:.4f} {t_loss: .4f} \t'
                      'Top1 Accuracy: {top1.val:.3f}% '.format(
                          epoch, batch_time=batch_time,
                        loss=losses, top1=top1s, t_loss=t_loss.float().item()))
                # print("   Epoch total time: {total_time:.3f}s".format(total_time=time.time()-start))

            if epoch % args.eval_step == 0 or epoch == args.epochs - 1:
                # top1 = validate_old(args, test_loader, model, criterion)
                test_model = avg_model if avg_model is not None else model
                top1 = validate(args, test_loader, test_model, criterion)
                is_best = top1 >= best_top1
                best_top1 = max(top1, best_top1)

                print(f"    Best top-1 acc: {best_top1:.2f}%")

                if epoch > 0 and epoch % args.save_every == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_top1': best_top1,
                        'first_optimizer': first_optimizer.state_dict(),
                        'second_optimizer': second_optimizer.state_dict(),
                        'avg_state_dict': avg_model.state_dict() if avg_model is not None else None
                    }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_top1': best_top1,
                        'first_optimizer': first_optimizer.state_dict(),
                        'second_optimizer': second_optimizer.state_dict(),
                        'avg_state_dict': avg_model.state_dict() if avg_model is not None else None
                    }, is_best, filename=os.path.join(args.save_dir, 'best.th'))

def train_basic(args, labeled_loader, test_loader, model, avg_model, criterion, first_optimizer, first_scheduler, first_scaler, best_top1):
    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    model.zero_grad()

    labeled_iter = iter(labeled_loader)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    
    start = time.time()
    end = time.time()

    for epoch in range(args.start_epoch, args.epochs):
            model.train()
            try:
                images_l, target = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_loader)
                images_l, target = next(labeled_iter)

            images_l = images_l.to(args.device)
            target = target.to(args.device)

            if args.ema > 0:
                avg_model.update_parameters(model)

            with amp.autocast(enabled=args.amp):
                output = model(images_l)
                t_loss = criterion(output, target)

            if args.amp:
                first_scaler.scale(t_loss).backward()
            else:
                t_loss.requires_grad_(True)
                t_loss.backward()

            if args.grad_clip > 0:
                first_scaler.unscale_(first_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.amp:
                first_scaler.step(first_optimizer)
                first_scaler.update()
            else:
                first_optimizer.step()

            first_scheduler.step()

            model.zero_grad()

            output = output.detach().float()
            loss = t_loss.float()
            # Measure accuracy and record loss
            top1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), target.size(0))
            top1s.update(top1.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % args.print_freq == 0 or epoch == args.epochs - 1:
                print('Epoch: [{0}]\t'
                      'Elapsed: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss: {loss.val:.4f} {t_loss: .4f} \t'
                      'Top1 Accuracy: {top1.val:.3f}% '.format(
                          epoch, batch_time=batch_time,
                        loss=losses, top1=top1s, t_loss=t_loss.float().item()))
                # print("   Epoch total time: {total_time:.3f}s".format(total_time=time.time()-start))

            if epoch % args.eval_step == 0 or epoch == args.epochs - 1:
                # top1 = validate_old(args, test_loader, model, criterion)
                test_model = avg_model if avg_model is not None else model
                top1 = validate(args, test_loader, test_model, criterion)
                is_best = top1 >= best_top1
                best_top1 = max(top1, best_top1)

                print(f"    Best top-1 acc: {best_top1:.2f}%")

                if epoch > 0 and epoch % args.save_every == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_top1': best_top1,
                        'first_optimizer': first_optimizer.state_dict(),
                        'avg_state_dict': avg_model.state_dict() if avg_model is not None else None
                    }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_top1': best_top1,
                        'first_optimizer': first_optimizer.state_dict(),
                        'avg_state_dict': avg_model.state_dict() if avg_model is not None else None
                    }, is_best, filename=os.path.join(args.save_dir, 'best.th'))

def main():
    args = parser.parse_args()
    best_top1 = 0
    model, avg_model, labeled_loader, unlabeled_loader, test_loader, criterion, first_optimizer, second_optimizer, first_scheduler, second_scheduler, first_scaler, second_scaler = initialize_components(args)
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            first_optimizer = checkpoint['first_optimizer']
            second_optimizer = checkpoint['second_optimizer']
            model.load_state_dict(checkpoint['state_dict'])
            print("checkpoint loaded'{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
            print('checkpoint epoch:', checkpoint['epoch'])
            print('resume lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            if avg_model is not None:
                    model_load_state_dict(avg_model, checkpoint['avg_state_dict'])
        else:
            print("no checkpoint found at '{}'".format(args.resume)) 
            return

    if args.evaluate:
        validate(test_loader, model, criterion)
        return
        
    train_loop(args, labeled_loader, unlabeled_loader, test_loader, model, avg_model, criterion, first_optimizer, second_optimizer, first_scheduler, second_scheduler, first_scaler, second_scaler, best_top1)
    return


def train(args, labeled_iter, unlabeled_iter, model, avg_model, criterion, first_optimizer, second_optimizer, first_scheduler, second_scheduler, first_scaler, second_scaler, moving_dot_product, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    model.train()
    start = time.time()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        target = target.to(args.device)
        target_var = target

        # compute output
        if args.mode == 'text':
            mask = input['attention_mask'].to(args.device)
            input_id = input['input_ids'].squeeze(1).to(args.device)
            output = model(input_id, mask)
        else:
            input_var = input.to(args.device)

            images_l = images_l.to(args.device)
            images_uw = images_uw.to(args.device)
            images_us = images_us.to(args.device)
            targets = targets.to(args.device)
            output = model(input_var)
        loss = criterion(output, target_var) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        top1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), target.size(0))
        top1s.update(top1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Elapsed: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top1 Accuracy: {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1s))
    print("Epoch total time: {total_time:.3f}s".format(total_time=time.time()-start))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.to(args.device)
            input_var = input.to(args.device)
            target_var = target.to(args.device)

            with amp.autocast(enabled=args.amp):
                # compute output
                if args.mode == 'text':
                    mask = input['attention_mask'].to(args.device)
                    input_id = input['input_ids'].squeeze(1).to(args.device)
                    output = model(input_id, mask)
                else:
                    input_var = input.to(args.device)
                    output = model(input_var)
                loss = criterion(output, target_var)
                output = output.float()
                loss = loss.float()

            # measure accuracy and record loss
            top1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), target.size(0))
            top1s.update(top1.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Validation Top1 Accuracy: {top1.avg:.3f}%'
          .format(top1=top1s))

    return top1s.avg

def validate_old(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1s = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.to(args.device)
            input_var = input.to(args.device)
            target_var = target.to(args.device)

            # compute output
            if args.mode == 'text':
                mask = input['attention_mask'].to(args.device)
                input_id = input['input_ids'].squeeze(1).to(args.device)
                output = model(input_id, mask)
            else:
                input_var = input.to(args.device)
                output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            top1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), target.size(0))
            top1s.update(top1.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Validation Top1 Accuracy: {top1.avg:.3f}%'
          .format(top1=top1s))

    return top1s.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def predict(args, input, model, transform=None):
    model.eval()
    if transform is None:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        input = transform(input)
    with torch.no_grad():
            if input.ndim == 3:
                input = input.unsqueeze(0)
            input_var = input.to(args.device)
            output = model(input_var)
        
    return output.cpu().numpy()



if __name__ == '__main__':
    main()
