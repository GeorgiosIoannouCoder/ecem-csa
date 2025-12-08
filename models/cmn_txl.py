import torch
import torch.nn as nn
from layers.transformer_xl import TransformerXLLayer, RelativePositionalEncoding

class CMN_TransformerXL(nn.Module):
    """
    将 Transformer-XL 的机制融入 CMN
    保留 CMN 的 speaker-specific memory 设计
    增强长距离上下文建模能力
    """
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']  # 300
        self.n_speakers = config['n_speakers']  # 2 (IEMOCAP 是双人对话)
        self.n_classes = config['n_classes']  # 6 (angry, happy, sad, neutral, excited, frustrated)
        self.mem_len = config['mem_len']  # 记忆长度，如 8
        self.n_layers = config['n_layers']  # Transformer 层数，如 2-4
        
        # === 输入编码 ===
        # 假设输入是文本特征 (可以是预训练的 BERT embeddings 或 GloVe)
        self.utterance_encoder = nn.GRU(
            input_size=config['input_dim'],  # 如 768 (BERT) 或 300 (GloVe)
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_proj = nn.Linear(self.d_model * 2, self.d_model)
        
        # === Transformer-XL 核心组件 ===
        self.pos_emb = RelativePositionalEncoding(self.d_model, max_len=512)
        
        # 为每个 speaker 分别建立 Transformer-XL 层
        # 这保留了 CMN 的 speaker-specific 思想
        self.speaker_layers = nn.ModuleDict({
            f'speaker_{i}': nn.ModuleList([
                TransformerXLLayer(
                    d_model=self.d_model,
                    n_heads=config['n_heads'],  # 如 8
                    d_ff=config['d_ff'],  # 如 1024
                    dropout=config['dropout']
                ) for _ in range(self.n_layers)
            ]) for i in range(self.n_speakers)
        })
        
        # === CMN 风格的跨说话者注意力 ===
        self.cross_speaker_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config['n_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # === 情感分类头 ===
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),  # *2 因为拼接了两个speaker的表示
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.d_model, self.n_classes)
        )
    
    def forward(self, utterances, speakers, lengths, memories=None):
        """
        Args:
            utterances: [batch, max_seq_len, max_utt_len, input_dim]
                       batch conversations, each with multiple utterances
            speakers: [batch, max_seq_len] speaker id for each utterance (0 or 1)
            lengths: [batch] actual sequence length for each conversation
            memories: dict of previous segment memories for each speaker
                     {'speaker_0': [batch, mem_len, d_model], 'speaker_1': ...}
        
        Returns:
            logits: [batch, max_seq_len, n_classes] emotion predictions
            new_memories: updated memories for next segment
        """
        batch_size, max_seq_len = speakers.shape
        
        # === Step 1: Encode each utterance ===
        utterance_features = []
        for t in range(max_seq_len):
            utt = utterances[:, t, :, :]  # [batch, max_utt_len, input_dim]
            # GRU encoding
            output, hidden = self.utterance_encoder(utt)  # output: [batch, max_utt_len, 2*d_model]
            # Take last hidden state
            utt_repr = self.encoder_proj(output[:, -1, :])  # [batch, d_model]
            utterance_features.append(utt_repr)
        
        utterance_features = torch.stack(utterance_features, dim=1)  # [batch, max_seq_len, d_model]
        
        # === Step 2: 按 speaker 分组并用 Transformer-XL 处理 ===
        # 准备相对位置编码
        mem_len = self.mem_len if memories is not None else 0
        total_len = max_seq_len + mem_len
        pos_emb = self.pos_emb(total_len)  # [total_len, d_model]
        
        # 存储每个 speaker 的表示
        speaker_outputs = {i: [] for i in range(self.n_speakers)}
        new_memories = {}
        
        for speaker_id in range(self.n_speakers):
            # 找出属于当前 speaker 的 utterances
            speaker_mask = (speakers == speaker_id)  # [batch, max_seq_len]
            
            # 提取对应的特征
            # 这里简化处理：我们仍然对整个序列建模，但加权关注特定speaker
            x = utterance_features.clone()  # [batch, max_seq_len, d_model]
            
            # 获取该 speaker 的记忆
            memory = memories.get(f'speaker_{speaker_id}') if memories else None
            
            # 通过多层 Transformer-XL
            for layer in self.speaker_layers[f'speaker_{speaker_id}']:
                x = layer(x, pos_emb, memory=memory)
                # 更新 memory: 使用当前 segment 的隐藏状态
                if memory is not None:
                    # 保留最近的 mem_len 个状态
                    memory = torch.cat([memory, x], dim=1)[:, -self.mem_len:, :]
                else:
                    memory = x[:, -self.mem_len:, :] if x.size(1) > self.mem_len else x
            
            speaker_outputs[speaker_id] = x  # [batch, max_seq_len, d_model]
            new_memories[f'speaker_{speaker_id}'] = memory.detach()  # 停止梯度传播
        
        # === Step 3: 跨 speaker 注意力 (CMN 的核心思想) ===
        # 让每个 speaker 的表示相互关注
        attended_outputs = []
        for speaker_id in range(self.n_speakers):
            other_id = 1 - speaker_id  # 对于双人对话
            
            query = speaker_outputs[speaker_id]
            key_value = speaker_outputs[other_id]
            
            # Cross attention
            attn_output, _ = self.cross_speaker_attn(
                query, key_value, key_value
            )  # [batch, max_seq_len, d_model]
            
            attended_outputs.append(attn_output)
        
        # === Step 4: 融合并分类 ===
        # 拼接两个 speaker 的表示
        combined = torch.cat(attended_outputs, dim=-1)  # [batch, max_seq_len, 2*d_model]
        
        # 情感分类
        logits = self.classifier(combined)  # [batch, max_seq_len, n_classes]
        
        return logits, new_memories
    
    def init_memory(self, batch_size, device):
        """初始化空记忆"""
        return {
            f'speaker_{i}': torch.zeros(
                batch_size, self.mem_len, self.d_model, device=device
            ) for i in range(self.n_speakers)
        }