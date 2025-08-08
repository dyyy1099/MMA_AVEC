import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader  # 复用测试数据加载器
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子确保结果可复现
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='测试已训练好的模型')
    parser.add_argument('--model-path', type=str, required=True,
                        help='best_model.pth的路径')
    parser.add_argument('--test-annotation', type=str, required=True,
                        help='新测试集的标注文件路径')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='测试时的批次大小')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载的工作进程数')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像的尺寸')
    parser.add_argument('--temporal-layers', type=int, default=1,
                        help='时间层的数量，需与训练时一致')
    return parser.parse_args()


def load_model(args):
    """加载模型结构并载入预训练权重"""
    # 创建与训练时相同的模型结构
    model = GenerateModel(args=args)
    model = torch.nn.DataParallel(model).cuda()

    # 加载模型权重
    checkpoint = torch.load(args.model_path)
    # 处理可能的键名不匹配问题
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # 设置为评估模式
    return model


def test(model, test_loader):
    """在测试集上运行模型并计算性能指标"""
    all_preds = []
    all_targets = []

    with torch.no_grad():  # 禁用梯度计算加速推理
        for images, target, audio in tqdm.tqdm(test_loader, desc="测试中"):
            # 数据移至GPU
            images = images.cuda()
            target = target.cuda().float()
            audio = audio.cuda()

            # 模型预测
            output = model(images, audio).squeeze()
            output = np.round(output.cpu().numpy())
            # 收集预测结果和真实标签
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)

    return rmse, mae, all_preds, all_targets


def main():
    args = parse_args()

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = load_model(args)

    # 准备测试数据
    print(f"加载测试数据: {args.test_annotation}")
    test_data = test_data_loader(
        list_file=args.test_annotation,
        num_segments=16,  # 需与训练时一致
        duration=1,  # 需与训练时一致
        image_size=args.img_size
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 执行测试
    rmse, mae, preds, targets = test(model, test_loader)

    # 输出结果
    print("\n测试结果:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # 可选：保存预测结果
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "predictions.npy"), preds)
    np.save(os.path.join(output_dir, "targets.npy"), targets)
    print(f"\n预测结果已保存至 {output_dir}")


if __name__ == '__main__':
    main()