import json
import matplotlib.pyplot as plt
import os

LOG_PATH = "KaydetmeNoktasi/train_logs.json"
SAVE_DIR = "KaydetmeNoktasi/Grafikler"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(LOG_PATH, "r") as f:
    logs = json.load(f)

epochs = list(range(1, len(logs['train_losses']) + 1))
def plot_graph(y_values, title, ylabel, filename):
    epochs = list(range(1, len(y_values) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, y_values, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

# === Grafikler ===
plot_graph(logs['train_losses'], "Eğitim Kaybı", "Train Loss", "train_loss.png")
plot_graph(logs['val_losses'], "Doğrulama Kaybı", "Val Loss", "val_loss.png")
plot_graph(logs['val_ious'], "Validation IoU", "IoU (%)", "val_iou.png")
plot_graph(logs['val_accs'], "Doğrulama Doğruluğu", "Accuracy (%)", "val_acc.png")
plot_graph(logs['val_f1s'], "F1 Skoru", "F1 (%)", "val_f1.png")
plot_graph(logs['lrs'], "Learning Rate Değişimi", "Learning Rate", "learning_rate.png")

print(f"Grafikler kaydedildi: {SAVE_DIR}/")
