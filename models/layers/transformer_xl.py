import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionalEncoding(nn.Module):
    """Transformer-XL 的相对位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 创建相对位置编码矩阵
        # R_{i-j} 表示位置 i 和 j 之间的相对距离
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len):
        """返回相对位置编码"""
        return self.pe[:seq_len]


class RelativeMultiHeadAttention(nn.Module):
    """
    Transformer-XL 的相对位置多头注意力
    实现论文中的公式：
    A_rel = q@k^T + q@R_{i-j}^T + u@k^T + v@R_{i-j}^T
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Query, Key, Value 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Transformer-XL 特有：分离的位置编码投影
        self.W_k_r = nn.Linear(d_model, d_model)
        
        # 可学习的全局偏置 u 和 v (论文中的 term c 和 d)
        self.u = nn.Parameter(torch.randn(self.n_heads, self.d_head))
        self.v = nn.Parameter(torch.randn(self.n_heads, self.d_head))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(self, x, pos_emb, memory=None, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model] 当前输入
            pos_emb: [seq_len + mem_len, d_model] 相对位置编码
            memory: [batch, mem_len, d_model] 前一个segment的记忆(可选)
            mask: attention mask
        """
        batch_size, seq_len, _ = x.shape
        
        # 如果有 memory，拼接到 key 和 value
        if memory is not None:
            # cat = concatenate along sequence dimension
            cat = torch.cat([memory, x], dim=1)  # [batch, mem_len+seq_len, d_model]
            mem_len = memory.size(1)
        else:
            cat = x
            mem_len = 0
        
        total_len = cat.size(1)
        
        # === 计算 Q, K, V ===
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(cat)  # [batch, total_len, d_model]
        v = self.W_v(cat)  # [batch, total_len, d_model]
        
        # Reshape for multi-head: [batch, n_heads, seq_len, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, total_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, total_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # === 计算位置编码的 key ===
        k_r = self.W_k_r(pos_emb)  # [total_len, d_model]
        k_r = k_r.view(total_len, self.n_heads, self.d_head)
        
        # === 计算注意力分数的四个项 (Transformer-XL 核心) ===
        # Term (a): content-based addressing
        AC = torch.einsum('bhid,bhjd->bhij', q, k)  # [batch, n_heads, seq_len, total_len]
        
        # Term (b): content-dependent positional bias
        BD = self._relative_shift(
            torch.einsum('bhid,jhd->bhij', q, k_r)  # [batch, n_heads, seq_len, total_len]
        )
        
        # Term (c): global content bias
        EF = torch.einsum('hd,bhjd->bhj', self.u, k).unsqueeze(2)  # [batch, n_heads, 1, total_len]
        
        # Term (d): global positional bias  
        GH = self._relative_shift(
            torch.einsum('hd,jhd->hj', self.v, k_r).unsqueeze(0).unsqueeze(2)
        )
        
        # 合并所有项
        attn_score = (AC + BD + EF + GH) * self.scale
        
        # 应用 mask
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + Dropout
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        
        # 计算输出
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_prob, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(attn_output)
        return output
    
    def _relative_shift(self, x):
        """
        实现相对位置的 shift 操作
        将 [batch, n_heads, seq_len, total_len] 转换为正确的相对位置索引
        """
        batch_size, n_heads, seq_len, total_len = x.shape
        
        # 添加一列零
        x = F.pad(x, (1, 0))  # [batch, n_heads, seq_len, total_len+1]
        
        # Reshape and slice
        x = x.view(batch_size, n_heads, total_len + 1, seq_len)
        x = x[:, :, 1:, :].view_as(x[:, :, :total_len, :])  # 去掉第一行
        
        return x


class TransformerXLLayer(nn.Module):
    """单层 Transformer-XL"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, pos_emb, memory=None, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            pos_emb: [total_len, d_model]
            memory: [batch, mem_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Multi-head attention with residual
        attn_out = self.attn(x, pos_emb, memory, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x