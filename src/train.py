import os
import time 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import *
from src.model import SimpleLSTM
from src.dataset import MaichartDataset, collate_fn

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """
    改进的模型训练函数，包含验证集监控、早停机制和学习率调度。
    
    Args:
        model (nn.Module): 要训练的 PyTorch 模型
        train_loader (DataLoader): 训练数据加载器，包含 (sequences, labels) 或 (sequences, labels, padding_mask)
        val_loader (DataLoader): 验证数据加载器，格式同训练数据加载器
        num_epochs (int, optional): 最大训练轮数。默认为 50
        learning_rate (float, optional): 初始学习率。默认为 0.001
    
    Returns:
        tuple[list, list]: 包含两个列表的元组
            - train_losses (list): 每个 epoch 的平均训练损失
            - val_losses (list): 每个 epoch 的平均验证损失
    
    Training Strategy:
        - Loss Function: MSE (均方误差) 用于回归任务
        - Optimizer: Adam 优化器，包含 L2 正则化 (weight_decay=1e-5)
        - Scheduler: ReduceLROnPlateau，验证损失停止改善时降低学习率
        - Early Stopping: 连续 10 个 epoch 验证损失无改善时停止训练
        - Gradient Clipping: 最大梯度范数限制为 1.0，防止梯度爆炸
    
    Model Checkpointing:
        - 自动保存验证损失最低的模型权重到 'best_model.pth'
        - 训练结束后可通过 model.load_state_dict(torch.load('best_model.pth')) 加载最佳模型
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    train_losses = []
    val_losses = []
    
    # 总训练开始时间
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batch_times = []
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - 训练阶段")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            batch_time = time.time() - batch_start_time
            train_batch_times.append(batch_time)
            train_loss += loss.item()

            # 每10个batch输出一次进度（可调整频率）
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_batch_time = np.mean(train_batch_times[-10:])  # 最近10个batch的平均时间
                print(f"  Batch [{batch_idx+1:4d}/{len(train_loader):4d}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Batch Time: {batch_time:.2f}s | "
                      f"Avg Time: {avg_batch_time:.2f}s | "
                      f"Seq Shape: {sequences.shape}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # === 验证阶段 ===
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - 验证阶段")
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
            print(f"早停计数器: {patience_counter}/{early_stop_patience}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 估算剩余时间
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        print(f"  已用时间: {elapsed_time/60:.1f}分钟 | 预计剩余: {estimated_remaining_time/60:.1f}分钟")
        print(f"{'='*60}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    
    total_training_time = time.time() - total_start_time
    print(f"\n🎉 训练完成!")
    print(f"总训练时间: {total_training_time/60:.1f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader):
    """全面评估模型性能"""
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
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
    accuracy_01 = np.mean(np.abs(predictions - true_values) <= 0.1)
    accuracy_02 = np.mean(np.abs(predictions - true_values) <= 0.2)
    accuracy_05 = np.mean(np.abs(predictions - true_values) <= 0.5)
    
    print(f"评估结果:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  ±0.1准确率: {accuracy_01:.3f}")
    print(f"  ±0.2准确率: {accuracy_02:.3f}")
    print(f"  ±0.5准确率: {accuracy_05:.3f}")
    
    return {
        'mse': mse, 'mae': mae, 'r2': r2,
        'acc_01': accuracy_01, 'acc_02': accuracy_02, 'acc_05': accuracy_05,
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

    # 实验配置
    config = {
        'model_type': 'SimpleLSTM',
        'input_size': 21,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers': 2,
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'early_stop_patience': 10
    }
    print(f"实验配置: {config}")
    print("="*80)

    # 创建数据集
    print("创建数据集...")
    dataset_start_time = time.time()
    train_dataset = MaichartDataset(SERIALIZED_DIR, TRAIN_DATA_PATH)
    test_dataset = MaichartDataset(SERIALIZED_DIR, TEST_DATA_PATH)
    dataset_time = time.time() - dataset_start_time
    print(f"数据集创建完成 ({dataset_time:.3f}s)")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    loader_start_time = time.time()

    # 使用分桶采样器创建训练数据加载器
    # 注意：使用 batch_sampler 时，DataLoader的 batch_size, shuffle, sampler, drop_last 参数必须为默认值
    # train_sampler = LevelIndexBucketSampler(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     drop_last=True # 在训练时丢弃不完整的batch通常是好的实践
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     collate_fn=collate_fn,
    #     num_workers=0   
    # )

    train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
    )

    # 测试加载器不需要分桶或打乱
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    loader_time = time.time() - loader_start_time
    print(f"数据加载器创建完成 ({loader_time:.3f}s)")
    print(f"测试批次数: {len(test_loader)}")

    # 创建模型
    print("\n创建模型...")
    model_start_time = time.time()
    model = SimpleLSTM(
        config['input_size'], 
        config['hidden_size'], 
        config['output_size'], 
        config['num_layers']
    )
    model_time = time.time() - model_start_time
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成 ({model_time:.3f}s)")
    print(f"模型参数数量: {param_count:,}")
    print(f"模型设备: {next(model.parameters()).device}")

    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate']
    )
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    print("\n=== 训练集评估 ===")
    train_results = evaluate_model(model, train_loader)
    print("\n=== 测试集评估 ===")
    test_results = evaluate_model(model, test_loader)

    # 保存实验记录
    log_file = save_experiment_log(model, train_losses, val_losses, train_results, test_results, config)


    return model, train_losses, val_losses, train_results, test_results
