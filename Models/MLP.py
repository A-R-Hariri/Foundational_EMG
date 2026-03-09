import torch
import torch.nn as nn; import torch.nn.functional as F

from utils import *


# ======== MODEL (EMBEDDING + PROJECTOR + CLASSIFIER HEAD) ========
class MLP(nn.Module):
    def __init__(self, feats, emb_dim=128, proj_dim=128, dropout=DROPOUT):
        super().__init__()

        self.fc1 = nn.Linear(feats, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_emb = nn.Linear(128, emb_dim)  # embedding
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # self.gelu = nn.GELU()

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
        if isinstance(m, (nn.Linear)):
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
        x /= 10.0

        x = self.relu(self.fc1(x))
        x = self.drop(x)

        x = self.relu(self.fc2(x))
        x = self.drop(x)

        x = self.relu(self.fc3(x))
        x = self.drop(x)

        emb = self.fc_emb(x)

        if return_proj:
            return self.proj(emb)
        if return_emb:
            return emb
        return self.classifier(emb)