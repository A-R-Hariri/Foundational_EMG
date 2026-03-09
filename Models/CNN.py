import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


# ======== MODEL (EMBEDDING + PROJECTOR + CLASSIFIER HEAD) ========
class CNN(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, 32, 8, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 6, padding="same")
        self.conv3 = nn.Conv1d(64, 128, 4, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # self.pool = nn.MaxPool1d(2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        with torch.no_grad():
            dummy = torch.zeros(1, ch, seq)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            fc_in = x.flatten(1).shape[1]

        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, emb_dim)  # embedding
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        # projector for SSL
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim))

        # classifier head (set later for fine-tune)
        self.classifier = None

        self.apply(self._init)
        print(f"Parameters count: {count_params(self):,}")

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def set_classifier(self, num_classes: int):
        self.classifier = nn.Sequential(
                        nn.Linear(self.fc_emb.out_features, 
                                self.fc_emb.out_features),
                        nn.ReLU(),
                        nn.Linear(self.fc_emb.out_features, num_classes))
    
    def set_linear_probe(self, num_classes: int):
        self.classifier = nn.Linear(self.fc_emb.out_features, num_classes)

    def forward(self, x, return_emb=False, return_proj=False):
        x *= 500.0

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)
        # x = self.pool(x)

        x = self.pool(x).squeeze(-1)
        # x = x.flatten(1)

        x = self.gelu(self.fc1(x))
        x = self.drop(x)
        emb = self.fc_emb(x)

        if return_proj:
            return self.proj(emb)
        if return_emb:
            return emb
        return self.classifier(emb)