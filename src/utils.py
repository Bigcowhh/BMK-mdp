import matplotlib.pyplot as plt
import numpy as np

def create_prediction_scatter_plot(predictions, true_values, title="Predictions vs True Values"):
    """创建预测值与真实值的散点图"""
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.6, s=20)
    
    # 添加完美预测线 (y=x)
    min_val = np.min([np.min(predictions), np.min(true_values)])
    max_val = np.max([np.max(predictions), np.max(true_values)])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 使用 plt.tight_layout() 确保布局紧凑
    plt.tight_layout()
    
    return fig
