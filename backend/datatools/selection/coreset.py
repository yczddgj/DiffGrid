from datetime import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from datatools.slice import Slice

import deepcore.datasets as datasets
import deepcore.nets as nets

from config import seeds, RESULT_DIR, RAW_DIR
from .utils import *
import config


class Args:
    def __init__(self):
        # 基础设置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = "CIFAR10"
        self.model = "ResNet18"
        self.selection = "Uniform"   # argparse 默认是 Uniform
        self.num_exp = 3
        self.num_eval = 10
        self.epochs = 200
        self.data_path = RAW_DIR
        self.gpu = None              # 可以是 list
        self.print_freq = 20
        self.fraction = 0.1
        self.seed = 0
        self.workers = 4
        self.cross = None            # 默认 None

        # 优化器 & 学习率调度
        self.optimizer = "SGD"
        self.lr = 0.1
        self.min_lr = 1e-4           # 原 argparse 默认
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = True
        self.scheduler = "CosineAnnealingLR"
        self.gamma = 0.5             # argparse 默认 0.5
        self.step_size = 50          # argparse 默认 50

        # batch 设置
        self.batch = 256             # 总的 batch
        self.train_batch = 256       # 若 None，则等于 batch
        self.selection_batch = 256   # 若 None，则等于 batch

        # 测试相关
        self.test_interval = 1       # argparse 默认
        self.test_fraction = 1.0

        # 路径
        self.save_path = RESULT_DIR
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

        # 数据集信息 (后续填充)
        self.channel = None
        self.im_size = None
        self.num_classes = None
        self.class_names = None


def train_from_slice(
    train_slice: str,
    eval_slice: str,
    model='ResNet18',
    seed: int = 0,
):
    """
    利用给定的子集进行训练
    """
    args = Args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = seed
    args.model = model
    if config.MODE != 'coreset':
        config.MODE = 'coreset'
    train_slice = Slice(train_slice)
    eval_slice = Slice(eval_slice)
    # 从 train_slice.name 格式中提取 dataset、selection、fraction 信息
    # 格式例如: CIFAR10-SES_0.01_2025-03-03 15:34:26.103515
    name_parts = train_slice.name.split('-')
    args.dataset = name_parts[0]
    selection_info = '-'.join(name_parts[1:]).split('_')
    args.selection = selection_info[0]
    args.fraction = float(selection_info[1])
    print(args.selection)
    # 将 slice 处理为 indices
    train_indices = train_slice.get_block_offset()
    eval_indices = eval_slice.get_block_offset()

    checkpoint = {}
    start_exp = 0
    start_epoch = 0
    single = False
    if args.seed != 0:
        args.num_exp = 1
        single = True
        print('单次训练')

    for exp in range(start_exp, args.num_exp):
        checkpoint = {}
        start_epoch = 0
        if not single:
            args.seed = seeds[exp]

        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_{fr}_epoch{epc}_exp{exp}_{dat}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=args.seed,
                                                                                         epc=args.epochs,
                                                                                         fr=args.fraction)

            os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
            log_file = open(os.path.join(args.save_path, 'logs',
                            checkpoint_name + 'log.txt'), 'w')
            original_stdout = sys.stdout
            sys.stdout = log_file

        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", length: ", len(
                  train_indices), ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ",   device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[
            args.dataset](args.data_path)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Augmentation
        if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            dst_train.transform = transforms.Compose(
                [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(), dst_train.transform])
        elif args.dataset == "ImageNet":
            dst_train.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        subset = train_indices
        val_indices = eval_indices

        dst_subset = torch.utils.data.Subset(
            dst_train, subset)

        _, _, _, _, _, _, dst_val, _ = datasets.__dict__[
            args.dataset](args.data_path)
        dst_val.data = dst_val.data[val_indices]
        dst_val.targets = dst_val.targets[val_indices]

        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dst_val, batch_size=args.train_batch, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)

        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        for model in models:
            if len(models) > 1:
                print("| Training on model %s" % model)

            network = nets.__dict__[model](
                channel, num_classes, im_size).to(args.device)

            if args.device == "cpu":
                print("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(
                    network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            if "state_dict" in checkpoint.keys():
                network.load_state_dict(checkpoint["state_dict"])

            criterion = nn.CrossEntropyLoss(
                reduction='none').to(args.device)

            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(
                    network.parameters(), args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            if args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                       eta_min=args.min_lr)
            elif args.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                            gamma=args.gamma)
            else:
                scheduler = torch.optim.lr_scheduler.__dict__[
                    args.scheduler](optimizer)
            scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

            if "opt_dict" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["opt_dict"])

            if "rec" in checkpoint.keys():
                rec = checkpoint["rec"]
            else:
                rec = init_recorder()

            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys(
            ) else 0.0

            if args.save_path != "":
                save_checkpoint({"exp": exp,
                                "subset": subset},
                                os.path.join(args.save_path, checkpoint_name + "unknown.ckpt"), 0, 0.)

            for epoch in range(start_epoch, args.epochs):
                train(train_loader, network, criterion, optimizer,
                      scheduler, epoch, args, rec, if_weighted=False)

                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network,
                                 criterion, epoch, args, rec)
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                            "epoch": epoch + 1,
                                             "state_dict": network.state_dict(),
                                             "opt_dict": optimizer.state_dict(),
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset},
                                            os.path.join(
                                                args.save_path, checkpoint_name + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " +
                  model if len(models) > 1 else "", end="\n\n")

            if args.save_path != "":
                test_acc = test_save(val_loader, network,
                                     criterion, epoch, args, rec)
                test_acc = np.where(test_acc == 0, -1, test_acc)
                print("validation acc: ", test_acc)
                np.save(os.path.join(args.save_path, 'tests', checkpoint_name +
                        "test_acc.npy"), test_acc)
                sys.stdout = original_stdout
                log_file.close()
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name +
                                     "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name + "%f.ckpt" % best_prec1))
                    os.rename(
                        os.path.join(args.save_path, 'logs', checkpoint_name +
                                     "log.txt"), os.path.join(args.save_path, 'logs', checkpoint_name + "%f.ckpt" % best_prec1) + ".log")
                except:
                    save_checkpoint({"exp": exp,
                                    "epoch": args.epochs,
                                     "state_dict": network.state_dict(),
                                     "opt_dict": optimizer.state_dict(),
                                     "best_acc1": best_prec1,
                                     "rec": rec,
                                     "subset": subset},
                                    os.path.join(args.save_path, checkpoint_name +
                                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)


from training_results import parse_results
