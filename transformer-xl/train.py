import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.cmn_txl import CMN_TransformerXL
from data.iemocap_loader import IEMOCAPDataset  
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def train():
    # === 配置 ===
    config = {
        'input_dim': 768,  # 如果用 BERT features
        'd_model': 300,
        'n_heads': 6,
        'd_ff': 1024,
        'n_layers': 3,
        'n_speakers': 2,
        'n_classes': 6,
        'mem_len': 8,  # 记忆长度：保留最近8个话语
        'dropout': 0.3,
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 50
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 加载数据 ===
    train_dataset = IEMOCAPDataset(split='train')
    val_dataset = IEMOCAPDataset(split='val')
    test_dataset = IEMOCAPDataset(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            collate_fn=test_dataset.collate_fn)
    
    # === 初始化模型 ===
    model = CMN_TransformerXL(config).to(device)
    
    # === 优化器和损失函数 ===
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for padding
    
    # === 训练循环 ===
    best_val_f1 = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            utterances = batch['utterances'].to(device)
            speakers = batch['speakers'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            # Forward pass
            # 初始化记忆 (第一个 segment)
            memories = model.init_memory(utterances.size(0), device)
            
            logits, _ = model(utterances, speakers, lengths, memories)
            
            # 计算损失
            # Flatten for cross entropy: [batch*seq_len, n_classes]
            logits_flat = logits.view(-1, config['n_classes'])
            labels_flat = labels.view(-1)
            
            loss = criterion(logits_flat, labels_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # === 验证 ===
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, config)
        
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_cmn_txl.pth')
            print(f'✓ Saved best model with F1: {val_f1:.4f}')
    
    # === 测试 ===
    model.load_state_dict(torch.load('best_model_cmn_txl.pth'))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, config)
    print(f'\n=== Test Results ===')
    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')


def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            utterances = batch['utterances'].to(device)
            speakers = batch['speakers'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            memories = model.init_memory(utterances.size(0), device)
            logits, _ = model(utterances, speakers, lengths, memories)
            
            # Loss
            logits_flat = logits.view(-1, config['n_classes'])
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            
            # 收集有效预测（排除 padding）
            for i in range(len(lengths)):
                valid_len = lengths[i]
                all_preds.extend(preds[i, :valid_len].cpu().numpy())
                all_labels.extend(labels[i, :valid_len].cpu().numpy())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, acc, f1


if __name__ == '__main__':
    train()