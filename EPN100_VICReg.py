import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch; print(torch.cuda.is_available())

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *
from models import CNN


MMAP_MODE = 'r'
# ======== LOAD ========
train_data = np.load(join(PATH, 'train_data.npy'), allow_pickle=True).item()
val_data = np.load(join(PATH, 'val_data.npy'), allow_pickle=True).item()
test_data = np.load(join(PATH, 'test_data.npy'), allow_pickle=True).item()

train_windows = np.load(join(PATH, 'train_windows.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(PATH, 'train_meta.npy'), allow_pickle=True).item()
val_windows = np.load(join(PATH, 'val_windows.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(PATH, 'val_meta.npy'), allow_pickle=True).item()
test_windows = np.load(join(PATH, 'test_windows.npy'), mmap_mode=MMAP_MODE)
test_meta = np.load(join(PATH, 'test_meta.npy'), allow_pickle=True).item()

train_windows.shape, val_windows.shape, test_windows.shape, \
    *[len(np.unique(meta['subjects'])) for meta in [train_meta, val_meta, test_meta]]


results = []
for SEED in [7, 7]:
    random.seed(SEED); np.random.seed(SEED)
    GENERATOR = torch.manual_seed(SEED)
    # ======== PIPELINE ========

    # ---- SSL subset (classes 0..11), labels unused ----
    ssl_x, _ = filter_by_classes(train_windows, train_meta["classes"], SSL_CLASSES)
    ssl_loader = create_ssl_loader(ssl_x, batch=BATCH_SIZE, shuffle=True)
    # ---- SSL subset (5 gestures only) ----
    ssl5_x, _ = filter_by_classes(train_windows, train_meta["classes"], FT_CLASSES)
    ssl5_loader = create_ssl_loader(ssl5_x, batch=BATCH_SIZE, shuffle=True)

    # ---- FT subset (classes 0..4) with relabel ----
    ft_train_x, ft_train_y = filter_by_classes(train_windows, train_meta["classes"], FT_CLASSES)
    ft_val_x, ft_val_y = filter_by_classes(val_windows, val_meta["classes"], FT_CLASSES)
    ft_test_x, ft_test_y = filter_by_classes(test_windows, test_meta["classes"], FT_CLASSES)

    ft_train_y = remap_labels(ft_train_y, FT_CLASSES)
    ft_val_y   = remap_labels(ft_val_y, FT_CLASSES)
    ft_test_y  = remap_labels(ft_test_y, FT_CLASSES)

    ft_train_loader = create_sup_loader(ft_train_x, ft_train_y, batch=BATCH_SIZE, shuffle=True)
    ft_val_loader = create_sup_loader(ft_val_x, ft_val_y, batch=BATCH_SIZE, shuffle=True)
    ft_test_loader = create_sup_loader(ft_test_x, ft_test_y, batch=BATCH_SIZE, shuffle=True)

    # ---- 12-class supervised loaders ----
    train_x, train_y = filter_by_classes(train_windows, train_meta["classes"], SSL_CLASSES)
    val_x,   val_y   = filter_by_classes(val_windows,   val_meta["classes"],   SSL_CLASSES)
    test_x,  test_y  = filter_by_classes(test_windows,  test_meta["classes"],  SSL_CLASSES)

    train_loader = create_sup_loader(train_x, train_y, batch=BATCH_SIZE, shuffle=True)
    val_loader = create_sup_loader(val_x, val_y, batch=BATCH_SIZE, shuffle=True)
    test_loader = create_sup_loader(test_x, test_y, batch=BATCH_SIZE, shuffle=True)

    # ---- class weights for FT ----
    ft_weights = compute_class_weight(class_weight="balanced", 
                classes=np.arange(len(FT_CLASSES)), y=ft_train_y).astype(np.float32)
    ft_weights = torch.from_numpy(ft_weights).to(DEVICE)
    ft_loss = nn.CrossEntropyLoss(weight=ft_weights)

    # ---- class weights for Total ----
    weights = compute_class_weight(class_weight="balanced", 
                classes=np.arange(len(SSL_CLASSES)), y=train_y).astype(np.float32)
    weights = torch.from_numpy(weights).to(DEVICE)
    loss = nn.CrossEntropyLoss(weight=weights)

    print(weights, ft_weights)

    # ---- BASE: Pretraining ----
    pretrained = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    pretrained = pretrain_vicreg(pretrained, ssl_loader, name=f"cnn_vicreg_ssl_seed{SEED}")

    pretrained_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    pretrained_5 = pretrain_vicreg(
        pretrained_5, ssl5_loader, name=f"cnn_vicreg_ssl5_seed{SEED}")

    # ---- EXP 1 ----
    model_1 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_1.load_state_dict(copy.deepcopy(pretrained.state_dict()))

    model_1.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_1.parameters():
        p.requires_grad = True

    model_1 = train_supervised(
        model_1, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained_then_ft_seed{SEED}",
        loss_fn=ft_loss,
    )
    acc_pt1, _, f1_pt1, bal_pt1 = evaluate_sup(model_1, ft_test_loader, ft_loss, DEVICE)

    # ---- EXP 1 (SSL-5): Pretrained on 5 gestures, full FT ----
    model_1_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_1_5.load_state_dict(copy.deepcopy(pretrained_5.state_dict()))

    model_1_5.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_1_5.parameters():
        p.requires_grad = True

    model_1_5 = train_supervised(
        model_1_5, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained5_then_ft_seed{SEED}",
        loss_fn=ft_loss,
    )
    acc_pt1_5, _, f1_pt1_5, bal_pt1_5 = evaluate_sup(
        model_1_5, ft_test_loader, ft_loss, DEVICE
    )

    # ---- EXP 2 ----
    model_2 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_2.load_state_dict(copy.deepcopy(pretrained.state_dict()))

    model_2.set_classifier(num_classes=len(FT_CLASSES))

    for p in model_2.parameters():
        p.requires_grad = True
    for p in model_2.conv1.parameters(): p.requires_grad = False
    for p in model_2.conv2.parameters(): p.requires_grad = False
    for p in model_2.conv3.parameters(): p.requires_grad = False

    model_2 = train_supervised(
        model_2, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained_frozen_cnn_then_ft_seed{SEED}",
        loss_fn=ft_loss,
    )
    acc_pt2, _, f1_pt2, bal_pt2 = evaluate_sup(model_2, ft_test_loader, ft_loss, DEVICE)

    # ---- EXP 2 (SSL-5): Frozen CNN, FT ----
    model_2_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_2_5.load_state_dict(copy.deepcopy(pretrained_5.state_dict()))

    model_2_5.set_classifier(num_classes=len(FT_CLASSES))

    for p in model_2_5.parameters():
        p.requires_grad = True
    for p in model_2_5.conv1.parameters(): p.requires_grad = False
    for p in model_2_5.conv2.parameters(): p.requires_grad = False
    for p in model_2_5.conv3.parameters(): p.requires_grad = False

    model_2_5 = train_supervised(
        model_2_5, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained5_frozen_cnn_then_ft_seed{SEED}",
        loss_fn=ft_loss,
    )
    acc_pt2_5, _, f1_pt2_5, bal_pt2_5 = evaluate_sup(
        model_2_5, ft_test_loader, ft_loss, DEVICE
    )

    # ---- EXP 2b: Fully frozen backbone (linear probe) ----
    model_lp = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_lp.load_state_dict(copy.deepcopy(pretrained.state_dict()))

    model_lp.set_linear_probe(num_classes=len(FT_CLASSES))

    for p in model_lp.parameters():
        p.requires_grad = False
    for p in model_lp.classifier.parameters():
        p.requires_grad = True

    model_lp = train_supervised(
        model_lp, ft_train_loader, ft_val_loader,
        name=f"cnn_linear_probe_seed{SEED}",
        loss_fn=ft_loss,
    )

    acc_lp, _, f1_lp, bal_lp = evaluate_sup(model_lp, ft_test_loader, ft_loss, DEVICE)

    # ---- EXP 2b (SSL-5): Linear probe ----
    model_lp_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_lp_5.load_state_dict(copy.deepcopy(pretrained_5.state_dict()))

    model_lp_5.set_linear_probe(num_classes=len(FT_CLASSES))

    for p in model_lp_5.parameters():
        p.requires_grad = False
    for p in model_lp_5.classifier.parameters():
        p.requires_grad = True

    model_lp_5 = train_supervised(
        model_lp_5, ft_train_loader, ft_val_loader,
        name=f"cnn_linear_probe_ssl5_seed{SEED}",
        loss_fn=ft_loss,
    )

    acc_lp_5, _, f1_lp_5, bal_lp_5 = evaluate_sup(
        model_lp_5, ft_test_loader, ft_loss, DEVICE
    )

    # ---- EXP 3 ----
    model_3 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)

    model_3.set_classifier(num_classes=len(FT_CLASSES))

    for p in model_3.parameters():
        p.requires_grad = True
        
    model_3 = train_supervised(
        model_3, ft_train_loader, ft_val_loader,
        name=f"cnn_raw_ft_only_seed{SEED}",
        loss_fn=ft_loss,
    )
    acc_raw, _, f1_raw, bal_raw = evaluate_sup(model_3, ft_test_loader, ft_loss, DEVICE)

    # ---- EXP 4 ----
    model_4 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)

    model_4.set_classifier(num_classes=len(SSL_CLASSES))

    for p in model_4.parameters():
        p.requires_grad = True
        
    model_4 = train_supervised(
        model_4, train_loader, val_loader,
        name=f"cnn_raw_total_seed{SEED}",
        loss_fn=loss,
    )
    acc_total, _, f1_total, bal_total = evaluate_sup(model_4, test_loader, loss, DEVICE)


    results.append({
        "seed": SEED,
        # ---- SSL pretrain (12) → FT (5) ----
        "ssl_ft": acc_pt1,
        "f1_ssl_ft": f1_pt1,
        "bal_ssl_ft": bal_pt1,
        # ---- SSL pretrain (12) → linear probe (5) ----
        "ssl_linear_probe": acc_lp,
        "f1_lp": f1_lp,
        "bal_lp": bal_lp,
        # ---- SSL pretrain (5) → FT (5) ----
        "ssl5_ft": acc_pt1_5,
        "f1_ssl5_ft": f1_pt1_5,
        "bal_ssl5_ft": bal_pt1_5,
        # ---- SSL pretrain (5) → FT frozen CNN (5) ----
        "ssl5_ft_frozen_cnn": acc_pt2_5,
        "f1_ssl5_ft_frozen_cnn": f1_pt2_5,
        "bal_ssl5_ft_frozen_cnn": bal_pt2_5,
        # ---- SSL pretrain (5) → linear probe (5) ----
        "ssl5_linear_probe": acc_lp_5,
        "f1_ssl5_lp": f1_lp_5,
        "bal_ssl5_lp": bal_lp_5,
        # ---- SSL pretrain (12) → FT frozen CNN (5) ----
        "ssl_ft_frozen_cnn": acc_pt2,
        "f1_ssl_ft_frozen_cnn": f1_pt2,
        "bal_ssl_ft_frozen_cnn": bal_pt2,
        # ---- raw supervised (5) ----
        "raw_5": acc_raw,
        "f1_raw_5": f1_raw,
        "bal_raw_5": bal_raw,
        # ---- raw supervised (12) ----
        "raw_12": acc_total,
        "f1_raw_12": f1_total,
        "bal_raw_12": bal_total,
    })

    SSL_EPOCHS = 50

df = pd.DataFrame(results)
# ---- summary ----
print(df.describe())
# ---- save full per-seed results ----
out_csv = "results_all_seeds.csv"
df.to_csv(out_csv, index=False)