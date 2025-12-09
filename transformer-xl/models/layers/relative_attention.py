"""
Transformer-XL 的相对位置注意力机制
包含高效的相对位置计算和注意力可视化工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RelativePositionBias(nn.Module):
    """
    相对位置偏置
    为 Transformer-XL 生成可学习的相对位置编码
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 可学习的相对位置嵌入
        # 索引 0 表示距离 0，索引 1 表示距离 1，依此类推
        self.relative_embeddings = nn.Parameter(
            torch.randn(max_len * 2 + 1, d_model) * 0.02
        )
    
    def forward(self, seq_len, mem_len=0):
        """
        生成相对位置嵌入矩阵
        
        Args:
            seq_len: 当前序列长度
            mem_len: 记忆长度
            
        Returns:
            [total_len, d_model] 相对位置嵌入
        """
        total_len = seq_len + mem_len
        
        # 对于位置 i 查询位置 j，相对距离是 i - j
        # 我们需要从 -mem_len 到 seq_len-1 的所有相对位置
        # 映射到 [0, max_len*2] 的索引
        positions = torch.arange(total_len, device=self.relative_embeddings.device)
        
        # 提取需要的嵌入
        # 添加 max_len 作为偏移，使负数索引变为正数
        indices = positions + self.max_len
        indices = indices.clamp(0, self.max_len * 2)
        
        return self.relative_embeddings[indices]


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码（Transformer 原始论文的方式）
    用于相对位置，不需要学习
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, seq_len):
        """返回 [seq_len, d_model] 的位置编码"""
        return self.pe[:seq_len]


class RelativeMultiHeadAttention(nn.Module):
    """
    Transformer-XL 的相对位置多头注意力
    
    实现公式（论文 Equation 3）：
    A_rel_i,j = q_i^T W_k^E x_j + q_i^T W_k^R R_{i-j} + u^T W_k^E x_j + v^T W_k^R R_{i-j}
    
    四个项的含义：
    (a) content-based addressing: 基于内容的注意力
    (b) content-dependent positional bias: 与内容相关的位置偏置
    (c) global content bias: 全局内容偏置
    (d) global positional bias: 全局位置偏置
    """
    def __init__(self, d_model, n_heads, dropout=0.1, use_learnable_pos=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Query, Key, Value 投影矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Transformer-XL 特有：分离的位置编码投影
        self.W_k_r = nn.Linear(d_model, d_model, bias=False)
        
        # 可学习的全局偏置（论文中的 u 和 v）
        self.u = nn.Parameter(torch.randn(n_heads, self.d_head))
        self.v = nn.Parameter(torch.randn(n_heads, self.d_head))
        
        # 位置编码生成器
        if use_learnable_pos:
            self.pos_encoder = RelativePositionBias(d_model)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 用于存储注意力权重（调试/可视化用）
        self.attn_weights = None
    
    def forward(self, x, memory=None, mask=None, return_attn=False):
        """
        Args:
            x: [batch, seq_len, d_model] 当前输入序列
            memory: [batch, mem_len, d_model] 前一个 segment 的记忆（可选）
            mask: [batch, seq_len, total_len] 注意力掩码（可选）
            return_attn: 是否返回注意力权重
        
        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, total_len] (如果 return_attn=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # === 1. 拼接 memory 和当前输入 ===
        if memory is not None and memory.size(1) > 0:
            cat = torch.cat([memory, x], dim=1)  # [batch, mem_len+seq_len, d_model]
            mem_len = memory.size(1)
        else:
            cat = x
            mem_len = 0
        
        total_len = cat.size(1)
        
        # === 2. 计算 Q, K, V ===
        q = self.W_q(x)      # [batch, seq_len, d_model]
        k = self.W_k(cat)    # [batch, total_len, d_model]
        v = self.W_v(cat)    # [batch, total_len, d_model]
        
        # Reshape 为多头: [batch, n_heads, seq_len/total_len, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, total_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, total_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # === 3. 生成相对位置编码 ===
        pos_emb = self.pos_encoder(total_len).to(x.device)  # [total_len, d_model]
        k_r = self.W_k_r(pos_emb)  # [total_len, d_model]
        k_r = k_r.view(total_len, self.n_heads, self.d_head)  # [total_len, n_heads, d_head]
        
        # === 4. 计算注意力分数的四个项 ===
        
        # Term (a): content-based addressing
        # q: [batch, n_heads, seq_len, d_head]
        # k: [batch, n_heads, total_len, d_head]
        # AC: [batch, n_heads, seq_len, total_len]
        AC = torch.einsum('bnid,bnjd->bnij', q, k)
        
        # Term (b): content-dependent positional bias
        # q: [batch, n_heads, seq_len, d_head]
        # k_r: [total_len, n_heads, d_head]
        # BD: [batch, n_heads, seq_len, total_len]
        BD = torch.einsum('bnid,jnd->bnij', q, k_r)
        BD = self._relative_shift(BD, mem_len)
        
        # Term (c): global content bias
        # u: [n_heads, d_head]
        # k: [batch, n_heads, total_len, d_head]
        # EF: [batch, n_heads, seq_len, total_len]
        EF = torch.einsum('nd,bnjd->bnj', self.u, k).unsqueeze(2)  # [batch, n_heads, 1, total_len]
        EF = EF.expand(-1, -1, seq_len, -1)
        
        # Term (d): global positional bias
        # v: [n_heads, d_head]
        # k_r: [total_len, n_heads, d_head]
        # GH: [n_heads, total_len]
        GH = torch.einsum('nd,jnd->nj', self.v, k_r)  # [n_heads, total_len]
        GH = GH.unsqueeze(0).unsqueeze(2)  # [1, n_heads, 1, total_len]
        GH = self._relative_shift(GH, mem_len)
        GH = GH.expand(batch_size, -1, seq_len, -1)
        
        # === 5. 合并所有项并缩放 ===
        attn_score = (AC + BD + EF + GH) * self.scale
        
        # === 6. 应用 mask ===
        if mask is not None:
            # mask: [batch, seq_len, total_len] 或 [batch, 1, seq_len, total_len]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, total_len]
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))
        
        # === 7. Softmax + Dropout ===
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        
        # 保存注意力权重（用于可视化）
        if return_attn:
            self.attn_weights = attn_prob.detach()
        
        # === 8. 计算输出 ===
        # attn_prob: [batch, n_heads, seq_len, total_len]
        # v: [batch, n_heads, total_len, d_head]
        # output: [batch, n_heads, seq_len, d_head]
        attn_output = torch.einsum('bnij,bnjd->bnid', attn_prob, v)
        
        # Reshape 回 [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        
        if return_attn:
            return output, self.attn_weights
        return output
    
    def _relative_shift(self, x, mem_len=0):
        """
        实现相对位置的 shift 操作
        将 [batch, n_heads, seq_len, total_len] 的注意力分数
        转换为正确的相对位置索引
        
        参考 Transformer-XL 论文 Appendix B
        """
        batch_size, n_heads, seq_len, total_len = x.shape
        
        # 添加一列零 padding
        x = F.pad(x, (1, 0))  # [batch, n_heads, seq_len, total_len+1]
        
        # Reshape 并切片
        x = x.view(batch_size, n_heads, total_len + 1, seq_len)
        
        # 移除第一行并恢复形状
        x = x[:, :, 1:, :].view(batch_size, n_heads, seq_len, total_len)
        
        # 如果有 memory，只保留有效的列
        if mem_len > 0:
            x = x[:, :, :, :total_len]
        
        return x


class MultiHeadAttentionWithRelPos(nn.Module):
    """
    带相对位置编码的多头注意力（简化版，用于对比实验）
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory=None, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            memory: [batch, mem_len, d_model]
            mask: attention mask
        
        Returns:
            [batch, seq_len, d_model]
        """
        attn_out = self.attn(x, memory, mask)
        x = self.norm(x + self.dropout(attn_out))
        return x


def test_relative_attention():
    """测试相对位置注意力的正确性"""
    print("=== Testing Relative Multi-Head Attention ===")
    
    batch_size = 2
    seq_len = 4
    mem_len = 3
    d_model = 64
    n_heads = 4
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    memory = torch.randn(batch_size, mem_len, d_model)
    
    # 初始化模型
    attn = RelativeMultiHeadAttention(d_model, n_heads)
    
    # Forward pass
    output, attn_weights = attn(x, memory, return_attn=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Memory shape: {memory.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len + mem_len), \
        "Attention weights shape mismatch!"
    
    # 检查注意力权重和为 1
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "Attention weights don't sum to 1!"
    
    print("✓ All tests passed!")
    
    return attn, output, attn_weights


if __name__ == '__main__':
    test_relative_attention()