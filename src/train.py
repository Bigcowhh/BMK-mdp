import os
import time 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from src.config import *
from src.config import TrainingConfig, DEFAULT_TRAINING_CONFIG
from src.model import SimpleLSTM
from src.dataset import MaichartDataset, collate_fn

def train_model(model, train_loader, val_loader, config: TrainingConfig = None, 
                experiment_name: Optional[str] = None, log_dir: Optional[str] = None) -> Tuple[List[float], List[float]]:
    """
    改进的模型训练函数，包含验证集监控、早停机制、学习率调度和 TensorBoard 日志记录。
    
    Args:
        model (nn.Module): 要训练的 PyTorch 模型
        train_loader (DataLoader): 训练数据加载器，包含 (sequences, labels) 或 (sequences, labels, padding_mask)
        val_loader (DataLoader): 验证数据加载器，格式同训练数据加载器
        config (TrainingConfig, optional): 训练配置对象。默认使用 DEFAULT_TRAINING_CONFIG
        experiment_name (str, optional): 实验名称，用于 TensorBoard 日志
        log_dir (str, optional): TensorBoard 日志目录
    
    Returns:
        tuple[list, list]: 包含两个列表的元组
            - train_losses (list): 每个 epoch 的平均训练损失
            - val_losses (list): 每个 epoch 的平均验证损失
    """
    if config is None:
        config = DEFAULT_TRAINING_CONFIG
    
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
    
    # 初始化 TensorBoard Writer
    writer = None
    if log_dir and experiment_name:
        tb_log_dir = os.path.join(log_dir, experiment_name)
        writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard 日志目录: {tb_log_dir}")
    
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
        if writer:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 记录模型参数分布
            for name, param in model.named_parameters():
                if param.grad is not None:                    
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
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
        if writer:
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
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
    
    # 关闭 TensorBoard Writer
    if writer:
        writer.close()
        print(f"TensorBoard 日志已保存到: {writer.log_dir}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, config: TrainingConfig = None, writer=None, epoch=None, phase=""):
    """全面评估模型性能，支持 TensorBoard 记录"""
    if config is None:
        config = DEFAULT_TRAINING_CONFIG
        
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
    
    # 记录到 TensorBoard
    if writer and epoch is not None:
        prefix = f"Metrics/{phase}" if phase else "Metrics"
        writer.add_scalar(f'{prefix}/MSE', mse, epoch)
        writer.add_scalar(f'{prefix}/MAE', mae, epoch)
        writer.add_scalar(f'{prefix}/R2', r2, epoch)
        
        # 记录准确率指标
        for threshold in accuracy_thresholds:
            acc_key = f'acc_{str(threshold).replace(".", "")}'
            threshold_str = str(threshold).replace(".", "_")
            writer.add_scalar(f'{prefix}/Accuracy_{threshold_str}', accuracy_results[acc_key], epoch)
        
        # 记录预测分布
        writer.add_histogram(f'{prefix}/Predictions', predictions, epoch)
        writer.add_histogram(f'{prefix}/True_Values', true_values, epoch)
        writer.add_histogram(f'{prefix}/Prediction_Errors', predictions - true_values, epoch)
        
        # 创建并记录散点图
        if len(predictions) <= config.max_scatter_points:  # 避免图像过于密集
            scatter_fig = create_prediction_scatter_plot(
                predictions, true_values, 
                title=f"Predictions vs True Values {phase}"
            )
            writer.add_figure(f'{prefix}/Prediction_Scatter', scatter_fig, epoch)
            plt.close(scatter_fig)  # 释放内存    
    return {
        'mse': mse, 'mae': mae, 'r2': r2,
        **accuracy_results,
        'predictions': predictions, 'true_values': true_values
    }

import json
import datetime
from datetime import datetime

def save_experiment_log(model, train_losses, val_losses, train_results, test_results, config):
    """保存实验记录"""
    
    # 创建实验日志目录
    log_dir = os.path.join(BASE_DIR, "experiment_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 实验配置和结果
    experiment_log = {
        'timestamp': timestamp,
        'model_config': config,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training': {
            'num_epochs': len(train_losses),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'train_losses': train_losses,
            'val_losses': val_losses
        },
        'evaluation': {
            'train_results': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                            for k, v in train_results.items() if k not in ['predictions', 'true_values']},
            'test_results': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                           for k, v in test_results.items() if k not in ['predictions', 'true_values']}
        }
    }
    
    # 保存日志
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    
    print(f"实验日志已保存: {log_file}")
    return log_file

def train_complete_pipeline():
    """完整的训练流程，包含 TensorBoard 日志记录"""

    # 实验配置
    config = TrainingConfig()
    print(f"实验配置: {config.to_dict()}")
    print("="*80)

    # 设置 TensorBoard 日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"experiment_{timestamp}"
    tb_log_dir = os.path.join(BASE_DIR, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    print(f"TensorBoard 实验名称: {experiment_name}")

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

    # 创建 TensorBoard Writer 并记录模型结构
    writer = SummaryWriter(os.path.join(tb_log_dir, experiment_name))
    
    # 记录模型结构图（使用一个样本数据）
    try:
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(DEVICE)
        writer.add_graph(model, sample_input)
        print("模型结构图已记录到 TensorBoard")
    except Exception as e:
        print(f"记录模型结构图失败: {e}")    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        config=config,
        experiment_name=experiment_name,
        log_dir=tb_log_dir
    )
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    print("\n=== 训练集评估 ===")
    train_results = evaluate_model(model, train_loader, config)
    print("\n=== 测试集评估 ===")
    test_results = evaluate_model(model, test_loader, config)# 保存实验记录
    log_file = save_experiment_log(model, train_losses, val_losses, train_results, test_results, config.to_dict())


    return model, train_losses, val_losses, train_results, test_results

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
