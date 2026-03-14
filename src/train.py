import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader ,Dataset
from typing import Optional

from copy import deepcopy

from src.evaluation import evaluate_model


class SentimentDataset(Dataset):
    def __init__(self,X,y):
        
        self.X=torch.tensor(X,dtype = torch.long)
        self.y=torch.tensor(y,dtype = torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self ,idx):
        return self.X[idx], self.y[idx]
    
class SentimentModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5
    ):
        super(SentimentModel, self).__init__()

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.attention_fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x):

        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        attention_scores = self.attention_fc(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        weighted_sum = (attention_weights * lstm_out).sum(dim=1)

        out = self.dropout(weighted_sum)
        out = self.fc(out)

        return out

def create_dataloaders(X_train ,y_train , X_test ,y_test , batch_size):
    
    train_dataset = SentimentDataset(X_train , y_train)
    test_dataset = SentimentDataset(X_test , y_test)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size , shuffle = False)
    
    return train_loader , test_loader

def train_one_epoch(model, dataloader, criterion, optimizer, device, gradient_clip: Optional[float] = None):
    """
    Train model for one epoch
    """
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        if gradient_clip is not None and gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

def train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    lr=0.001,
    weight_decay=0.0,
    device=None,
    early_stop_patience: Optional[int] = 3,
    min_delta: float = 0.0,
    lr_scheduler_patience: Optional[int] = 2,
    gradient_clip: Optional[float] = 1.0,
    class_weights: Optional[torch.Tensor] = None,
):
    """
    Full training loop
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if lr_scheduler_patience is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=lr_scheduler_patience,
            min_lr=1e-6,
            verbose=True,
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=gradient_clip)
       
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)
        improved = val_loss + min_delta < best_val_loss
        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if early_stop_patience is not None and no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered (no improvement for {early_stop_patience} epochs).")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            break

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

def save_model(model, path="models/sentiment_model.pth"):
    """
    Save PyTorch model weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
    print(f"Model saved to: {path}")
    
