import os
import time 
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List
from src.config import *
from src.config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from src.logger import Logger
from src.model import SimpleLSTM
from src.dataset import MaichartDataset, collate_fn

def train_model(model, train_loader, val_loader, config: TrainingConfig, logger: Logger) -> Tuple[List[float], List[float]]:
    """
    改进的模型训练函数，包含验证集监控、早停机制、学习率调度和日志记录。
    
    Args:
        model (nn.Module): 要训练的 PyTorch 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        config (TrainingConfig): 训练配置对象
        logger (Logger): 用于记录日志的 Logger 实例
    
    Returns:
        tuple[list, list]: 包含训练和验证损失的元组
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        config.scheduler_mode, 
        patience=config.scheduler_patience, 
        factor=config.scheduler_factor
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    # 总训练开始时间
    total_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batch_times = []
        print(f"Epoch [{epoch+1}/{config.num_epochs}] - 训练阶段")
        print("-" * 60)
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 将数据移动到 GPU
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
            
            optimizer.step()
            batch_time = time.time() - batch_start_time
            train_batch_times.append(batch_time)
            train_loss += loss.item()            # 每10个batch输出一次进度（可调整频率）
            if (batch_idx + 1) % config.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_batch_time = np.mean(train_batch_times[-10:])  # 最近10个batch的平均时间
                print(f"  Batch [{batch_idx+1:4d}/{len(train_loader):4d}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Batch Time: {batch_time:.2f}s | "
                      f"Avg Time: {avg_batch_time:.2f}s | "
                      f"Seq Shape: {sequences.shape}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 记录训练损失到 TensorBoard
        logger.log_scalar('Loss/Train', avg_train_loss, epoch)
        logger.log_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录模型参数分布
        for name, param in model.named_parameters():
            if param.grad is not None:
                logger.log_histogram(f'Parameters/{name}', param, epoch)
                logger.log_histogram(f'Gradients/{name}', param.grad, epoch)

        # === 验证阶段 ===
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}] - 验证阶段")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                # 将数据移动到 GPU
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 记录验证损失到 TensorBoard
        logger.log_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            model_save_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"新的最佳模型已保存 (验证损失: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{config.early_stop_patience}")
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 估算剩余时间
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = config.num_epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        print(f"  已用时间: {elapsed_time/60:.1f}分钟 | 预计剩余: {estimated_remaining_time/60:.1f}分钟")
        print(f"{'='*60}")
        
        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_training_time = time.time() - total_start_time
    print(f"\n🎉 训练完成!")
    print(f"总训练时间: {total_training_time/60:.1f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, config: TrainingConfig, logger: Logger = None, epoch: int = None, phase: str = ""):
    """全面评估模型性能，支持 TensorBoard 记录"""
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy().flatten())
            true_values.extend(labels.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # 计算各种指标
    mse = np.mean((predictions - true_values) ** 2)
    mae = np.mean(np.abs(predictions - true_values))
    r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
    
    # 准确性分析（在不同误差范围内的比例）
    accuracy_results = {}
    accuracy_thresholds = config.accuracy_thresholds
    for threshold in accuracy_thresholds:
        accuracy_results[f'acc_{str(threshold).replace(".", "")}'] = np.mean(np.abs(predictions - true_values) <= threshold)    
    print(f"评估结果 {phase}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    for threshold in accuracy_thresholds:
        acc_key = f'acc_{str(threshold).replace(".", "")}'
        print(f"  ±{threshold}准确率: {accuracy_results[acc_key]:.3f}")
    
    results = {
        'mse': mse, 'mae': mae, 'r2': r2,
        **accuracy_results,
        'predictions': np.array(predictions), 
        'true_values': np.array(true_values)
    }

    # 记录到 TensorBoard
    if logger and epoch is not None:
        logger.log_evaluation_metrics(results, epoch, phase)
        logger.log_evaluation_histograms(results, epoch, phase)
        
        # 创建并记录散点图
        if len(predictions) <= config.max_scatter_points:
            scatter_fig = create_prediction_scatter_plot(
                predictions, true_values, 
                title=f"Predictions vs True Values {phase}"
            )
            logger.log_scatter_plot(scatter_fig, epoch, phase)
            plt.close(scatter_fig)
            
    return results

def train_complete_pipeline():
    """完整的训练流程，包含 TensorBoard 日志记录"""

    # 实验配置
    config = TrainingConfig()
    print(f"实验配置: {config.to_dict()}")
    print("="*80)

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
        num_workers=0
    )

    # 测试加载器不需要分桶或打乱
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    loader_time = time.time() - loader_start_time
    print(f"数据加载器创建完成 ({loader_time:.3f}s)")
    print(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")    # 创建模型
    print("\n创建模型...")
    model_start_time = time.time()
    model = SimpleLSTM(
        config.input_size, 
        config.hidden_size, 
        config.output_size, 
        config.num_layers
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
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        config=config,
        logger=logger
    )
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    
    print("\n=== 训练集评估 ===")
    train_results = evaluate_model(model, train_loader, config, logger, len(train_losses), "Train")
    
    print("\n=== 测试集评估 ===")
    test_results = evaluate_model(model, test_loader, config, logger, len(train_losses), "Test")

    # 保存实验记录
    logger.save_experiment_summary(model, config, train_losses, val_losses, train_results, test_results)

    # 关闭 logger
    logger.close()

    return

def create_prediction_scatter_plot(predictions, true_values, title="Predictions vs True Values"):
    """创建预测值与真实值的散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, s=20)
    
    # 添加完美预测线 (y=x)
    min_val = min(min(predictions), min(true_values))
    max_val = max(max(predictions), max(true_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 使用 plt.tight_layout() 确保布局紧凑
    plt.tight_layout()
    
    return plt.gcf()
