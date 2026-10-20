import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datatools.trainer.base import BaseTrainer
from datatools.trainer.registry import register_trainer


@register_trainer("classification")
class ClassificationTrainer(BaseTrainer):
    def __init__(self, model_name: str = "resnet18", batch_size: int = 32, epochs: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()
        self._init_data()

    def _init_model(self):
        self.model = getattr(models, self.model_name)(pretrained=False)
        self.model.fc = nn.Linear(
            self.model.fc.in_features, 1000)  # 默认1000类，可调整
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_data(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.train_slice.transform = transform
        self.test_slice.transform = transform if self.test_slice else None
        self.train_loader = DataLoader(
            self.train_slice, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_slice, batch_size=self.batch_size) if self.test_slice else None

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    def evaluate(self):
        if not self.test_loader:
            print("No test set provided.")
            return
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Test Accuracy: {acc:.2%}")

    def save(self):
        path = f"{self.output_dir}/model_{self.model_name}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
