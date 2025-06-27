import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    统一管理 TensorBoard 和文件日志的类。
    """
    def __init__(self, base_dir, experiment_name_prefix="experiment"):
        """
        初始化 Logger。

        Args:
            base_dir (str): 项目根目录或日志存储的根目录。
            experiment_name_prefix (str): 实验名称前缀。
        """
        # 1. 创建唯一的实验名称和时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"{experiment_name_prefix}_{self.timestamp}"
        
        # 2. 定义并创建 TensorBoard 和实验日志的目录
        self.tb_log_dir = os.path.join(base_dir, "tensorboard_logs", self.experiment_name)
        self.exp_log_dir = os.path.join(base_dir, "experiment_logs")
        os.makedirs(self.tb_log_dir, exist_ok=True)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # 3. 初始化 SummaryWriter
        self.writer = SummaryWriter(self.tb_log_dir)
        print(f"Logger initialized. TensorBoard logs at: {self.tb_log_dir}")

    def log_scalar(self, tag, scalar_value, global_step):
        """记录标量值"""
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_histogram(self, tag, values, global_step):
        """记录直方图"""
        self.writer.add_histogram(tag, values, global_step)

    def log_model_graph(self, model, input_to_model):
        """记录模型结构图"""
        try:
            self.writer.add_graph(model, input_to_model)
            print("模型结构图已记录到 TensorBoard")
        except Exception as e:
            print(f"记录模型结构图失败: {e}")

    def log_evaluation_metrics(self, metrics, epoch, phase=""):
        """记录评估指标到 TensorBoard"""
        prefix = f"Metrics/{phase}" if phase else "Metrics"
        self.log_scalar(f'{prefix}/MSE', metrics['mse'], epoch)
        self.log_scalar(f'{prefix}/MAE', metrics['mae'], epoch)
        self.log_scalar(f'{prefix}/R2', metrics['r2'], epoch)
        
        for key, value in metrics.items():
            if key.startswith('acc_'):
                threshold_str = key.replace('acc_', '')
                self.log_scalar(f'{prefix}/Accuracy_{threshold_str}', value, epoch)

    def log_evaluation_histograms(self, results, epoch, phase=""):
        """记录评估相关的直方图"""
        prefix = f"Metrics/{phase}" if phase else "Metrics"
        self.log_histogram(f'{prefix}/Predictions', results['predictions'], epoch)
        self.log_histogram(f'{prefix}/True_Values', results['true_values'], epoch)
        self.log_histogram(f'{prefix}/Prediction_Errors', results['predictions'] - results['true_values'], epoch)

    def log_scatter_plot(self, fig, epoch, phase=""):
        """记录散点图"""
        prefix = f"Metrics/{phase}" if phase else "Metrics"
        self.writer.add_figure(f'{prefix}/Prediction_Scatter', fig, epoch)

    def save_experiment_summary(self, model, config, train_losses, val_losses, train_results, test_results):
        """
        保存完整的实验日志到 JSON 文件。
        """
        experiment_log = {
            'timestamp': self.timestamp,
            'experiment_name': self.experiment_name,
            'model_config': config.to_dict(),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'training': {
                'num_epochs': len(train_losses),
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'best_val_loss': min(val_losses) if val_losses else None,
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
        
        log_file = os.path.join(self.exp_log_dir, f"{self.experiment_name}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)
        print(f"实验日志已保存: {log_file}")
        return log_file

    def close(self):
        """关闭 SummaryWriter"""
        self.writer.close()
        print("Logger closed.")
