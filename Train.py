import os
import torch
import torch.nn as nn
import time
import numpy as np
import json


def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    labels = labels.float()

    inter = (preds * labels).sum()
    union = ((preds + labels) >= 1).float().sum()
    iou = (inter / (union + 1e-6)).item() * 100

    tp = inter
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = (2 * precision * recall / (precision + recall + 1e-6)).item() * 100

    acc = (preds == labels).float().mean().item() * 100
    return {'IoU': iou, 'F1': f1, 'Accuracy': acc}


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    start = time.time()
    total = len(dataloader)

    for idx, (inputs, labels) in enumerate(dataloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start
            eta = elapsed / idx * (total - idx)
            print(f"   Batch {idx}/{total} | Loss: {loss:.4f} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    avg_loss = running_loss / total
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(logits_cat, labels_cat)
    print(f"[Train] Loss: {avg_loss:.4f} | IoU: {metrics['IoU']:.2f}% | F1: {metrics['F1']:.2f}% | Acc: {metrics['Accuracy']:.2f}%")
    return avg_loss, metrics


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += loss_fn(logits, labels).item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(logits_cat, labels_cat)
    return avg_loss, metrics, logits_cat, labels_cat


def threshold_search(logits, labels):
    print("\n Threshold araması (IoU):")
    best_thresh = 0.5
    best_iou = 0.0
    for t in np.arange(0.1, 0.86, 0.05):
        metrics = compute_metrics(logits, labels, threshold=t)
        print(f"   - Threshold {t:.2f} → IoU: {metrics['IoU']:.2f}%")
        if metrics['IoU'] > best_iou:
            best_iou = metrics['IoU']
            best_thresh = t
    print(f" En iyi threshold: {best_thresh:.2f} (IoU: {best_iou:.2f}%)")
    return best_thresh


def train_model(model, dataset, device, epochs=100, batch_size=4, val_split=0.2, save_dir="KaydetmeNoktasi"):
    print(f"\n Eğitim Başladı - Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Model Parametre Sayısı: {sum(p.numel() for p in model.parameters())}\n")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    pos_weight = torch.tensor([5.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    model.to(device)

    os.makedirs(save_dir, exist_ok=True)
    best_iou = 0.0
    patience = 7
    stop_counter = 0

    #  Eğitim loglarını tutmak için listeler
    train_losses, val_losses = [], []
    val_ious, val_accs, val_f1s = [], [], []
    lrs_log = []  #  Learning Rate log

    for epoch in range(1, epochs + 1):
        print(f"\n Epoch {epoch}/{epochs} ------------------------")
        train_loss, train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_metrics, val_logits, val_labels = validate_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f"📊 Epoch {epoch} Özeti: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        print(f"     Val IoU: {val_metrics['IoU']:.2f}%, Acc: {val_metrics['Accuracy']:.2f}%, F1: {val_metrics['F1']:.2f}%")

        #  Logları listeye ekle
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_metrics['IoU'])
        val_accs.append(val_metrics['Accuracy'])
        val_f1s.append(val_metrics['F1'])
        lrs_log.append(optimizer.param_groups[0]['lr'])  #  Öğrenme hızı kaydı

        #  Her epoch sonunda JSON dosyasına yaz
        logs = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_ious": val_ious,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
            "lrs": lrs_log
        }
        with open(os.path.join(save_dir, "train_logs.json"), "w") as f:
            json.dump(logs, f)

        #  Epoch modeli kaydet
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

        #  En iyi modeli kaydet
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save(model.state_dict(), os.path.join(save_dir, "en_iyi_model.pth"))
            print(f"💾 En iyi model kaydedildi (Val IoU: {best_iou:.2f}%)")
            stop_counter = 0
        else:
            stop_counter += 1
            print(f"⏳ early stop sayacı: {stop_counter}/{patience}")

        if stop_counter >= patience:
            print(" Early stopping devreye girdi!")
            break

    #  Threshold araması
    best_threshold = threshold_search(val_logits, val_labels)

    print(f"\n Eğitim tamamlandı. Log ve model kaydedildi.\n")
