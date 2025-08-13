import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Dataset import FloodDataset
from DeepLabV3Plus import DeepLabV3Plus


MODEL_PATH = "KaydetmeNoktasi/en_iyi_model.pth"
BASE_DIR = "."
SAVE_DIR = "SunumGorselleri"
NUM_SAMPLES = 15
THRESHOLD = 0.1
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(in_channels=8, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Source 1 (SAR) ve Source 2 görüntüler yükleniyor
# çünkü model 8 kanal bekliyor, ama sadece SAR etiketlerine göre görselleştiriyoruz
dataset = FloodDataset(
    base_dir=BASE_DIR,
    use_s1_labels=True,    #  sadece SAR maskesi
    use_s2_labels=False,   #  optik maskeleri alma
    use_s2=True            # optik görüntüler modele gidecek
)

def stretch_percentile(img, pmin=2, pmax=98):
    vmin, vmax = np.percentile(img, (pmin, pmax))
    return np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)

saved = 0
for i in range(len(dataset)):
    x, y = dataset[i]  # x: (8,H,W), y: (1,H,W)
    gt = y[0].numpy()

    if gt.sum() < 200:
        continue  # çok az flood varsa atla

    # S1 VV bandı 0. kanal olabilir (emin değilsen print(x.shape))
    vv = x[0].numpy()
    vv_norm = stretch_percentile(vv)

    with torch.no_grad():
        out = model(x.unsqueeze(0).to(device))  # tüm 8 kanallı girişi veriyoruz
        pred = torch.sigmoid(out)[0, 0].cpu().numpy()
        pred_mask = (pred > THRESHOLD).astype(float)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axs[0].imshow(vv_norm, cmap="gray")
    axs[0].set_title("Giriş (S1 VV)")
    axs[0].axis("off")

    axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Gerçek Maske")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("Tahmin Maske")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sunum_{saved + 1}.png"))
    plt.close()

    saved += 1
    if saved >= NUM_SAMPLES:
        break

print(f" {saved} adet kaliteli görsel kaydedildi: {SAVE_DIR}/ klasörüne")
