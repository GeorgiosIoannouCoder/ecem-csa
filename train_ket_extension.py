# train_ket_extension.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from ket_wrapper import KETOptimizedERCModel
# from your_project.models import CMNEncoder, TransformerXLEncoder, EACLEncoder


def dummy_collate(batch):
    """
    Placeholder collate_fn. In your actual code, this should:
        - pad input_ids
        - pad attention_mask
        - align knowledge_features
        - return labels
    """
    # Replace with your real collate_fn
    raise NotImplementedError


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)               # [B, T]
        attention_mask = batch["attention_mask"].to(device)     # [B, T]
        knowledge_features = batch["knowledge_features"].to(device)  # [B, T, K]
        labels = batch["labels"].to(device)                     # [B]

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            knowledge_features=knowledge_features,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Init your base encoder from teammates' code ----
    # Example:
    # base_encoder = CMNEncoder(...)
    # or:
    # base_encoder = TransformerXLEncoder(...)
    # or:
    # base_encoder = EACLEncoder(...)
    base_encoder = ...  # TODO: plug your model here

    hidden_dim = 256       # must match base_encoder hidden size
    knowledge_dim = 64     # your ConceptNet + VAD feature dim
    num_labels = 7         # e.g., # of emotion classes

    model = KETOptimizedERCModel(
        base_encoder=base_encoder,
        hidden_dim=hidden_dim,
        knowledge_dim=knowledge_dim,
        num_labels=num_labels,
        dropout=0.1,
        cls_pooling="cls",  # or "mean"
    ).to(device)

    # ---- 2. Prepare data ----
    train_dataset = ...  # your dataset that includes knowledge_features
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=dummy_collate,  # replace with your real one
    )

    # ---- 3. Optimizer ----
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # ---- 4. Training loop ----
    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}")

    # ---- 5. Save model for your GitHub repo ----
    torch.save(model.state_dict(), "ket_optimized_model.pt")


if __name__ == "__main__":
    main()
