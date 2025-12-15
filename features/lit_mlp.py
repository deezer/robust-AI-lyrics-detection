import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitMLP(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # optim
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


class TwoStreamLitMLP(pl.LightningModule):
    def __init__(self, embedding_sizes):
        super().__init__()
        self.embedding_sizes = embedding_sizes

        # Create separate projection networks for each embedding type
        self.projection_nets = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.Linear(size, 256),
                    nn.ReLU(),
                    nn.Linear(size, 128),
                    nn.ReLU(),
                )
                for size in embedding_sizes
            ]
        )

        # Combined size after concatenation
        combined_size = 128 * len(embedding_sizes)

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 128), nn.ReLU(), nn.Linear(128, 2)
        )

        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def forward(self, x):
        # Split input tensor along embedding dimensions
        start_idx = 0
        projected_embeddings = []

        for i, size in enumerate(self.embedding_sizes):
            end_idx = start_idx + size
            embedding = x[:, start_idx:end_idx]
            projected = self.projection_nets[i](embedding)
            projected_embeddings.append(projected)
            start_idx = end_idx

        # Concatenate projected embeddings
        combined = torch.cat(projected_embeddings, dim=1)

        # Final classification
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
