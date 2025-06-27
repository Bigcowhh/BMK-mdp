
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional

from src.config import TrainingConfig, DEVICE
from src.logger import Logger
from src.utils import create_prediction_scatter_plot

def evaluate_model(model, data_loader, config: TrainingConfig, logger: Optional[Logger] = None, epoch: Optional[int] = None, phase: str = ""):
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
