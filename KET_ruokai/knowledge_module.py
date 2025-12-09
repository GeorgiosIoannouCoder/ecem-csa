# knowledge_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeFusion(nn.Module):
    """
    KET-style knowledge injection module.

    Inputs:
        hidden_states: [batch, seq_len, hidden_dim]
            Contextual token representations from your base model
            (CMN / Transformer-XL / EACL encoder).
        knowledge_features: [batch, seq_len, knowledge_dim]
            Per-token external knowledge features
            (e.g., ConceptNet embeddings + NRC-VAD features).
        attention_mask: [batch, seq_len] (optional)
            1 for valid tokens, 0 for padding.

    Output:
        enriched_states: [batch, seq_len, hidden_dim]
            Knowledge-augmented hidden states.
    """

    def __init__(
        self,
        hidden_dim: int,
        knowledge_dim: int,
        fusion: str = "gated_add",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.knowledge_dim = knowledge_dim
        self.fusion = fusion

        # Project knowledge features into same space as hidden states
        self.knowledge_proj = nn.Linear(knowledge_dim, hidden_dim)

        # Gating network to control how much knowledge to inject
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, knowledge_features, attention_mask=None):
        """
        hidden_states: [B, T, H]
        knowledge_features: [B, T, K]
        attention_mask: [B, T] or None
        """
        if knowledge_features is None:
            # Fallback: no knowledge → return original states
            return hidden_states

        # Project knowledge into hidden_dim
        k_proj = self.knowledge_proj(knowledge_features)  # [B, T, H]

        # Simple gating: decide per token how much to mix text vs knowledge
        fusion_input = torch.cat([hidden_states, k_proj], dim=-1)  # [B, T, 2H]
        gate_logits = self.gate(fusion_input)                      # [B, T, H]
        gate = torch.sigmoid(gate_logits)                          # [B, T, H]

        if self.fusion == "gated_add":
            fused = hidden_states + gate * k_proj
        elif self.fusion == "concat":
            # If you want concat, you’ll need a downstream proj; by default use add.
            fused = self.layer_norm(self.dropout(
                torch.tanh(self.knowledge_proj(knowledge_features))
            ))
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        # Optional masking (just to be safe)
        if attention_mask is not None:
            # attention_mask: [B, T] → [B, T, 1]
            mask = attention_mask.unsqueeze(-1).type_as(fused)
            fused = fused * mask

        # Residual + layer norm
        enriched_states = self.layer_norm(hidden_states + self.dropout(fused))
        return enriched_states
