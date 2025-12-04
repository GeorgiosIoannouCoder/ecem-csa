# ket_wrapper.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from knowledge_module import KnowledgeFusion


class KETOptimizedERCModel(nn.Module):
    """
    Wrapper that adds KET-style knowledge fusion on top of an existing encoder.

    base_encoder is expected to return a dict or object with:
        - hidden_states or last_hidden_state: [B, T, H]
    You can adapt this to your actual CMN / Transformer-XL / EACL implementations.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        hidden_dim: int,
        knowledge_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        cls_pooling: str = "cls",  # or "mean"
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.cls_pooling = cls_pooling

        self.knowledge_fusion = KnowledgeFusion(
            hidden_dim=hidden_dim,
            knowledge_dim=knowledge_dim,
            fusion="gated_add",
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def _pool(self, hidden_states, attention_mask):
        """
        Get utterance-level representation from token-level states.
        """
        if self.cls_pooling == "cls":
            # Assume first token is [CLS] or equivalent
            return hidden_states[:, 0, :]  # [B, H]
        elif self.cls_pooling == "mean":
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # [B, T, 1]
            summed = (hidden_states * mask).sum(dim=1)                  # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)                     # [B, 1]
            return summed / denom
        else:
            raise ValueError(f"Unknown pooling: {self.cls_pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        knowledge_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **encoder_kwargs: Dict[str, Any],
    ):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        knowledge_features: [B, T, K] (aligned with tokens)
        labels: [B] (emotion labels)
        """

        # ---- 1. Run base encoder (your teammates' model) ----
        encoder_outputs = self.base_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )

        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs.get(
                "last_hidden_state",
                encoder_outputs.get("hidden_states"),
            )
        else:
            # e.g., HuggingFace returns a model output object
            hidden_states = encoder_outputs.last_hidden_state

        # hidden_states: [B, T, H]
        # ---- 2. Inject external knowledge (KET-style) ----
        enriched_states = self.knowledge_fusion(
            hidden_states=hidden_states,
            knowledge_features=knowledge_features,
            attention_mask=attention_mask,
        )

        # ---- 3. Pool & classify ----
        pooled = self._pool(enriched_states, attention_mask)  # [B, H]
        logits = self.classifier(self.dropout(pooled))        # [B, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "enriched_states": enriched_states,
            "raw_hidden_states": hidden_states,
        }
