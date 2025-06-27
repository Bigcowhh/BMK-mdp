import os
import time
import torch
from torch.utils.data import DataLoader

from src.config import (
    TrainingConfig,
    BASE_DIR,
    SERIALIZED_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_DIR,
    DEVICE,
)
from src.logger import Logger
from src.model import SimpleLSTM
from src.dataset import MaichartDataset, collate_fn
from src.trainer import Trainer
from src.evaluation import evaluate_model


def train_complete_pipeline():
    """完整的训练流程，包含 TensorBoard 日志记录"""

    # 实验配置
    config = TrainingConfig()
    print(f"实验配置: {config.to_dict()}")
    print("=" * 80)

    # 初始化 Logger
    logger = Logger(BASE_DIR, experiment_name_prefix="maimai_difficulty_prediction")

    # 创建数据集
    print("\n创建数据集...")
    dataset_start_time = time.time()
    train_dataset = MaichartDataset(SERIALIZED_DIR, TRAIN_DATA_PATH)
    test_dataset = MaichartDataset(SERIALIZED_DIR, TEST_DATA_PATH)
    dataset_time = time.time() - dataset_start_time
    print(f"数据集创建完成 ({dataset_time:.3f}s)")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    loader_start_time = time.time()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    loader_time = time.time() - loader_start_time
    print(f"数据加载器创建完成 ({loader_time:.3f}s)")
    print(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")

    # 创建模型
    print("\n创建模型...")
    model_start_time = time.time()
    model = SimpleLSTM(
        config.input_size,
        config.hidden_size,
        config.output_size,
        config.num_layers,
    )
    model_time = time.time() - model_start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成 ({model_time:.3f}s)")
    print(f"模型参数数量: {param_count:,}")
    print(f"模型设备: {next(model.parameters()).device}")

    # 记录模型结构图
    try:
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(DEVICE)
        logger.log_model_graph(model, sample_input)
    except Exception as e:
        print(f"记录模型结构图失败: {e}")

    # 训练模型
    print("\n开始训练...")
    trainer = Trainer(model, train_loader, test_loader, config, logger)
    train_losses, val_losses = trainer.train()

    # 加载最佳模型并评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))

    print("\n=== 训练集评估 ===")
    train_results = evaluate_model(
        model, train_loader, config, logger, len(train_losses), "Train"
    )

    print("\n=== 测试集评估 ===")
    test_results = evaluate_model(
        model, test_loader, config, logger, len(train_losses), "Test"
    )

    # 保存实验记录
    logger.save_experiment_summary(
        model, config, train_losses, val_losses, train_results, test_results
    )

    # 关闭 logger
    logger.close()

    return
