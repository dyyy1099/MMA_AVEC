from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
import tqdm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int)

    parser.add_argument('--exper-name', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()
    return args


def main(set, args):
    data_set = set + 1

    # 日志路径设置

    print(f"*********** AVEC Dataset Fold {data_set} ***********")
    log_dir = f'./log/AVEC-{time_str}-log'
    log_txt_path = f'{log_dir}/log.txt'
    log_curve_path = f'{log_dir}/log.png'
    checkpoint_path = f'{log_dir}/checkpoint/model.pth'
    train_annotation_file_path = f"./annotation/train_labels.txt"
    test_annotation_file_path = f"./annotation/dev_labels.txt"

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/checkpoint', exist_ok=True)
    os.makedirs(f'{log_dir}/code', exist_ok=True)
    # 复制代码文件（保持原有逻辑）
    code_files = ['main.py', 'train.sh',main.py
                  'models/Generate_Model.py', 'models/Temporal_Model.py',
                  'dataloader/video_dataloader.py', 'dataloader/video_transform.py',
                  'models/models_vit.py', 'AudioMAE/audio_models_vit.py']
    for f in code_files:
        dst = f'{log_dir}/code/{f}'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(f, dst)
    best_rmse = float('inf')
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)
    model = GenerateModel(args=args)

    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = True #False

    for name, param in model.named_parameters():
        if "our_classifier" in name:
            param.requires_grad = True
        if "positional_embedding" in name:
            param.requires_grad = True
        if "learnable_prompts" in name:
            param.requires_grad = True
        if "pos_embed" in name:
            param.requires_grad = True
        if "audio_proj" in name:
            param.requires_grad = True
        if "temporal" in name:
            param.requires_grad = True
        if "gate" in name:
            param.requires_grad = True
        if "context_att" in name:
            param.requires_grad = True
        if "learnable_q" in name:
            param.requires_grad = True
        if "audio_att" in name:
            param.requires_grad = True
        if "norm_xt" in name:
            param.requires_grad = True
        if "norm_xt_2" in name:
            param.requires_grad = True
        if "norm_qs" in name:
            param.requires_grad = True


    model = torch.nn.DataParallel(model).cuda()

    # print params
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')

    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')


    # 打印可训练参数
    print('************************ 可训练参数 ************************')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print('************************************************************')

    # 保存参数配置
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')

    # 回归任务：使用均方误差损失
    criterion = nn.MSELoss().cuda()

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cudnn.benchmark = True

    # 数据加载
    train_data = train_data_loader(
        list_file=train_annotation_file_path,
        num_segments=16,
        duration=1,
        image_size=args.img_size,
        args=args
    )
    test_data = test_data_loader(
        list_file=test_annotation_file_path,
        num_segments=16,
        duration=1,
        image_size=args.img_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    for epoch in range(args.epochs):
        inf = f'******************** Epoch {epoch} ********************'
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(f'{inf}\n')
            f.write(f'Current learning rate: {lr}\n')
        print(inf)
        print(f'Current learning rate: {lr}')

        # 训练一个epoch
        train_rmse, train_mae, train_loss = train(
            train_loader, model, criterion, optimizer, epoch, args, log_txt_path
        )

        # 验证
        val_rmse, val_mae, val_loss = validate(
            val_loader, model, criterion, args, log_txt_path
        )

        scheduler.step()

        # 保存最优模型（RMSE越小越好）
        is_best = val_rmse < best_rmse
        best_rmse = min(val_rmse, best_rmse)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_rmse': best_rmse,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder
        }, is_best, checkpoint_path)

        # 更新记录并绘图
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_loss, train_rmse, val_loss, val_rmse)
        recorder.plot_curve(log_curve_path)

        print(f'The best RMSE: {best_rmse:.4f}')
        print(f'An epoch time: {epoch_time:.2f}s')
        with open(log_txt_path, 'a') as f:
            f.write(f'The best RMSE: {best_rmse:.4f}\n')
            f.write(f'An epoch time: {epoch_time:.2f}s\n')

    # 最终评估
    final_rmse, final_mae = compute_final_metrics(val_loader, model, checkpoint_path)
    return final_rmse, final_mae


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    rmse_meter = AverageMeter('RMSE', ':.4f')
    mae_meter = AverageMeter('MAE', ':.4f')
    progress = ProgressMeter(
        len(train_loader), [losses, rmse_meter, mae_meter],
        prefix=f"Epoch: [{epoch}]",
        log_txt_path=log_txt_path
    )

    model.train()
    all_preds = []
    all_targets = []

    for i, (images, target, audio) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda().float()  # 回归目标为浮点数
        audio = audio.cuda()

        # 模型预测
        output = model(images, audio).squeeze()  # 形状: (batch_size,)
        loss = criterion(output, target)

        # 计算指标
        preds = output.detach().cpu().numpy()
        targets = target.cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

        # 实时计算批次指标（用于进度显示）
        batch_rmse = np.sqrt(mean_squared_error(targets, preds))
        batch_mae = mean_absolute_error(targets, preds)

        # 更新指标记录
        losses.update(loss.item(), images.size(0))
        rmse_meter.update(batch_rmse, images.size(0))
        mae_meter.update(batch_mae, images.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        if i % args.print_freq == 0:
            progress.display(i)

    # 计算整个epoch的指标
    epoch_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    epoch_mae = mean_absolute_error(all_targets, all_preds)
    return epoch_rmse, epoch_mae, losses.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    rmse_meter = AverageMeter('RMSE', ':.4f')
    mae_meter = AverageMeter('MAE', ':.4f')
    progress = ProgressMeter(
        len(val_loader), [losses, rmse_meter, mae_meter],
        prefix='Validation: ',
        log_txt_path=log_txt_path
    )

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target, audio) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda().float()
            audio = audio.cuda()

            output = model(images, audio).squeeze()
            loss = criterion(output, target)

            # 收集预测和目标值
            preds = output.cpu().numpy()
            targets = target.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

            # 批次指标
            batch_rmse = np.sqrt(mean_squared_error(targets, preds))
            batch_mae = mean_absolute_error(targets, preds)

            # 更新指标
            losses.update(loss.item(), images.size(0))
            rmse_meter.update(batch_rmse, images.size(0))
            mae_meter.update(batch_mae, images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

    # 计算整个验证集的指标
    epoch_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    epoch_mae = mean_absolute_error(all_targets, all_preds)

    print(f'Validation RMSE: {epoch_rmse:.4f}, MAE: {epoch_mae:.4f}')
    with open(log_txt_path, 'a') as f:
        f.write(f'Validation RMSE: {epoch_rmse:.4f}, MAE: {epoch_mae:.4f}\n')
    return epoch_rmse, epoch_mae, losses.avg


def compute_final_metrics(val_loader, model, checkpoint_path):
    """加载最优模型计算最终指标"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, target, audio in tqdm.tqdm(val_loader):
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            output = model(images, audio).squeeze()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    print(f'Final RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return rmse, mae


def save_checkpoint(state, is_best, checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, f'{os.path.dirname(checkpoint_path)}/best_model.pth')


class AverageMeter(object):
    """计算并存储指标的平均值和当前值"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """进度显示工具"""

    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class RecorderMeter(object):
    """记录并绘制训练曲线（损失和RMSE）"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [train, val]
        self.epoch_rmse = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [train, val]

    def update(self, idx, train_loss, train_rmse, val_loss, val_rmse):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_rmse[idx, 0] = train_rmse
        self.epoch_rmse[idx, 1] = val_rmse
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        """绘制损失和RMSE曲线"""
        title = 'Train/Validation Loss and RMSE'
        dpi = 80
        width, height = 1600, 800
        figsize = (width / dpi, height / dpi)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # 绘制损失曲线
        ax1.set_title('Loss Curve', fontsize=14)
        ax1.plot(range(self.total_epoch), self.epoch_losses[:, 0], 'g-', label='Train Loss')
        ax1.plot(range(self.total_epoch), self.epoch_losses[:, 1], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 绘制RMSE曲线
        ax2.set_title('RMSE Curve', fontsize=14)
        ax2.plot(range(self.total_epoch), self.epoch_rmse[:, 0], 'b-', label='Train RMSE')
        ax2.plot(range(self.total_epoch), self.epoch_rmse[:, 1], 'm-', label='Val RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    if args.exper_name:
        time_str += f'-{args.exper_name}'

    print('************************ 参数配置 ************************')
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('**********************************************************')

    all_fold = 1
    # 遍历所有折叠
    total_rmse = 0.0  # 累计所有折叠的RMSE
    total_mae = 0.0  # 累计所有折叠的MAE
    for set in range(all_fold):
        rmse, mae = main(set, args)
        total_rmse += rmse
        total_mae += mae

    print('********* 最终结果 *********')
    print(f'平均RMSE: {total_rmse / all_fold:.4f}')
    print(f'平均MAE: {total_mae / all_fold:.4f}')
    print('****************************')