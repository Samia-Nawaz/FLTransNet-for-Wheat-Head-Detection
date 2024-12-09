import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super(TransformerFeatureExtractor, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MSFB(nn.Module):
    def __init__(self, in_channels):
        super(MSFB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        return out1 + out3 + out5

class SAB(nn.Module):
    def __init__(self, embed_dim):
        super(SAB, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4)

    def forward(self, x):
        # Reshape for attention [Batch, Seq, Features]
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)  # [Seq, Batch, Features]
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Reshape back
        return x

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            for i in range(num_layers)
        ])
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = F.relu(layer(torch.cat(features, dim=1)))
            features.append(out)
        return self.lff(torch.cat(features, dim=1))

class FLTransNet(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, depth=6, growth_rate=64, num_layers=4):
        super(FLTransNet, self).__init__()

        # Transformer for feature extraction
        self.transformer = TransformerFeatureExtractor(embed_dim, num_heads, depth)

        # MSFB Block
        self.msfb = MSFB(in_channels=embed_dim)

        # SAB Block
        self.sab = SAB(embed_dim=embed_dim)

        # RDB Block
        self.rdb = RDB(in_channels=embed_dim, growth_rate=growth_rate, num_layers=num_layers)

        # Final layers for labels and bounding boxes
        self.fc_labels = nn.Linear(embed_dim, 10)  # Example for 10 classes
        self.fc_bboxes = nn.Linear(embed_dim, 4)  # For bounding box [x, y, w, h]

    def forward(self, x):
        # Transformer Feature Extraction
        x = self.transformer(x)

        # Multi-Scale Feature Block
        x = x.permute(0, 2, 1).view(x.size(0), -1, int(x.size(1) ** 0.5), int(x.size(1) ** 0.5))  # Reshape to [B, C, H, W]
        x = self.msfb(x)

        # Self-Attention Block
        x = self.sab(x)

        # Residual Dense Block
        x = self.rdb(x)

        # Global average pooling
        x = torch.mean(x, dim=[2, 3])  # [Batch, Channels]

        # Labels and Bounding Boxes
        labels = self.fc_labels(x)
        bboxes = self.fc_bboxes(x)

        return labels, bboxes

# Data Preparation
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.FakeData(transform=transform)
    test_dataset = datasets.FakeData(transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training and Evaluation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FLTransNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_labels = nn.CrossEntropyLoss()
    criterion_bboxes = nn.MSELoss()

    train_loader, test_loader = load_data()

    epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            labels, bboxes = model(images)

            # Simulated targets for demonstration purposes
            targets_labels = torch.randint(0, 10, (images.size(0),)).to(device)
            targets_bboxes = torch.rand((images.size(0), 4)).to(device)

            loss_labels = criterion_labels(labels, targets_labels)
            loss_bboxes = criterion_bboxes(bboxes, targets_bboxes)
            loss = loss_labels + loss_bboxes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Testing and Metrics
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            labels, bboxes = model(images)

            targets_labels = torch.randint(0, 10, (images.size(0),)).to(device)

            preds = labels.argmax(dim=1).cpu().numpy()
            targets = targets_labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot()
    plt.show()

    # Loss Plot
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Over Epochs')
    plt.show()
