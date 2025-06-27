import time
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Tuple, List

from src.config import TrainingConfig, DEVICE, MODEL_DIR
from src.logger import Logger

class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig, logger: Logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            config.scheduler_mode,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        train_batch_times = []
        print(f"Epoch [{epoch+1}/{self.config.num_epochs}] - 训练阶段")
        print("-" * 60)

        for batch_idx, (sequences, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_max_norm)
            self.optimizer.step()

            batch_time = time.time() - batch_start_time
            train_batch_times.append(batch_time)
            train_loss += loss.item()

            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == len(self.train_loader):
                avg_batch_time = np.mean(train_batch_times[-10:])
                print(f"  Batch [{batch_idx+1:4d}/{len(self.train_loader):4d}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Batch Time: {batch_time:.2f}s | "
                      f"Avg Time: {avg_batch_time:.2f}s | "
                      f"Seq Shape: {sequences.shape}")

        avg_train_loss = train_loss / len(self.train_loader)
        self.logger.log_scalar('Loss/Train', avg_train_loss, epoch)
        self.logger.log_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.logger.log_histogram(f'Parameters/{name}', param, epoch)
                self.logger.log_histogram(f'Gradients/{name}', param.grad, epoch)
        
        return avg_train_loss

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}] - 验证阶段")
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.logger.log_scalar('Loss/Validation', avg_val_loss, epoch)
        return avg_val_loss

    def _check_early_stopping(self, avg_val_loss: float) -> bool:
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            model_save_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"新的最佳模型已保存 (验证损失: {self.best_val_loss:.6f})")
        else:
            self.patience_counter += 1
            print(f"早停计数器: {self.patience_counter}/{self.config.early_stop_patience}")
        
        return self.patience_counter >= self.config.early_stop_patience

    def train(self) -> Tuple[List[float], List[float]]:
        train_losses = []
        val_losses = []
        total_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            avg_train_loss = self._train_epoch(epoch)
            avg_val_loss = self._validate_epoch(epoch)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            self.scheduler.step(avg_val_loss)

            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            elapsed_time = time.time() - total_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.config.num_epochs - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            print(f"  已用时间: {elapsed_time/60:.1f}分钟 | 预计剩余: {estimated_remaining_time/60:.1f}分钟")
            print(f"{'='*60}")

            if self._check_early_stopping(avg_val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_training_time = time.time() - total_start_time
        print(f"\n🎉 训练完成!")
        print(f"总训练时间: {total_training_time/60:.1f} 分钟")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")

        return train_losses, val_losses
