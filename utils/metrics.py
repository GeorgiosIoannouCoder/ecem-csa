"""
评估指标计算
包含准确率、F1、混淆矩阵等
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionMetrics:
    """情感识别任务的评估指标"""
    
    def __init__(self, emotion_labels=None):
        """
        Args:
            emotion_labels: 情感类别名称列表
                          如 ['angry', 'happy', 'sad', 'neutral']
        """
        if emotion_labels is None:
            # IEMOCAP 4-class 默认标签
            self.emotion_labels = ['angry', 'happy', 'sad', 'neutral']
        else:
            self.emotion_labels = emotion_labels
        
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.all_preds = []
        self.all_labels = []
    
    def update(self, preds, labels, mask=None):
        """
        更新预测结果
        
        Args:
            preds: [batch, seq_len] 或 [batch*seq_len] 预测标签
            labels: [batch, seq_len] 或 [batch*seq_len] 真实标签
            mask: [batch, seq_len] 或 [batch*seq_len] 有效位置掩码（可选）
        """
        # 转为 numpy
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # Flatten
        preds = preds.flatten()
        labels = labels.flatten()
        
        # 过滤无效位置（padding 或 -1）
        if mask is not None:
            mask = mask.flatten().astype(bool)
            preds = preds[mask]
            labels = labels[mask]
        else:
            # 移除标签为 -1 的位置
            valid_idx = labels >= 0
            preds = preds[valid_idx]
            labels = labels[valid_idx]
        
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())
    
    def compute(self):
        """
        计算所有指标
        
        Returns:
            dict: 包含 accuracy, weighted_f1, macro_f1 等指标
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        if len(preds) == 0:
            return {
                'accuracy': 0.0,
                'weighted_f1': 0.0,
                'macro_f1': 0.0,
                'weighted_precision': 0.0,
                'weighted_recall': 0.0
            }
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0),
            'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
            'weighted_precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'weighted_recall': recall_score(labels, preds, average='weighted', zero_division=0),
        }
        
        # 每个类别的 F1
        class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, label in enumerate(self.emotion_labels):
            if i < len(class_f1):
                metrics[f'f1_{label}'] = class_f1[i]
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        获取混淆矩阵
        
        Returns:
            numpy.ndarray: [n_classes, n_classes]
        """
        if len(self.all_preds) == 0:
            return np.zeros((len(self.emotion_labels), len(self.emotion_labels)))
        
        return confusion_matrix(
            self.all_labels, 
            self.all_preds,
            labels=list(range(len(self.emotion_labels)))
        )
    
    def get_classification_report(self):
        """
        获取详细分类报告
        
        Returns:
            str: sklearn 的分类报告
        """
        if len(self.all_preds) == 0:
            return "No predictions available"
        
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.emotion_labels,
            zero_division=0
        )
    
    def plot_confusion_matrix(self, save_path=None, normalize=False):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 保存路径（可选）
            normalize: 是否归一化
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def compute_metrics(preds, labels, mask=None):
    """
    快速计算指标（不需要累积）
    
    Args:
        preds: [batch, seq_len] 预测
        labels: [batch, seq_len] 标签
        mask: [batch, seq_len] 掩码
    
    Returns:
        dict: 指标字典
    """
    metrics = EmotionMetrics()
    metrics.update(preds, labels, mask)
    return metrics.compute()


class AverageMeter:
    """用于跟踪平均值和当前值"""
    def __init__(self):
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
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, mode='max', delta=0.0):
        """
        Args:
            patience: 容忍的 epoch 数
            mode: 'max' 表示越大越好，'min' 表示越小越好
            delta: 最小改进量
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        """
        Args:
            score: 当前指标值
            epoch: 当前 epoch
        
        Returns:
            bool: 是否是最佳模型
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def test_metrics():
    """测试评估指标"""
    print("=== Testing Metrics ===")
    
    # 模拟预测和标签
    preds = torch.tensor([[0, 1, 2, 3], [1, 1, 2, 0]])
    labels = torch.tensor([[0, 1, 2, 2], [1, 1, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])  # 最后一个位置是 padding
    
    # 计算指标
    metrics = EmotionMetrics(['angry', 'happy', 'sad', 'neutral'])
    metrics.update(preds, labels, mask)
    
    results = metrics.compute()
    print("\nMetrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nClassification Report:")
    print(metrics.get_classification_report())
    
    print("\nConfusion Matrix:")
    print(metrics.get_confusion_matrix())
    
    print("\n✓ Metrics test passed!")


if __name__ == '__main__':
    test_metrics()