import torch
import numpy as np
import os
import csv
import time
from DeepLabV3Plus import DeepLabV3Plus
from Dataset import FloodDataset


class FloodStats:
    def __init__(self,
                 model_path="KaydetmeNoktasi/en_iyi_model.pth",
                 base_dir=".",
                 threshold=None,  # threshold artık opsiyonel
                 in_channels=8,
                 out_channels=1,
                 batch_size=4,
                 device=None):

        self.model_path = model_path
        self.base_dir = base_dir
        self.threshold = threshold  # threshold boşsa otomatik belirlenecek
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model = DeepLabV3Plus(in_channels=in_channels, out_channels=out_channels)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Dataset yükle (augmentations kapalı)
        self.dataset = FloodDataset(base_dir=base_dir,
                                    use_s1_labels=True,
                                    use_s2_labels=False,
                                    use_s2=True)
        self.dataset.augment = lambda **kwargs: kwargs

        print(f"📂 Dataset uzunluğu: {len(self.dataset)} örnek bulundu.")
        if len(self.dataset) == 0:
            print("⚠️ Dataset boş! base_dir yolunu kontrol etmelisin.")

        self.piksel_alani = 10 * 10
        self.su_yuksekligi = 1

        # Eğer threshold verilmediyse, otomatik hesapla
        if self.threshold is None:
            self.threshold = self.find_best_threshold()
            print(f"🔍 Otomatik seçilen threshold: {self.threshold:.2f}")

    def find_best_threshold(self):
        """Validation set üzerinde basit bir threshold taraması yapar."""
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, shuffle=True)
        inputs, _ = next(iter(loader))
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.sigmoid(logits)

        best_t, best_mean = 0.3, 1.0
        for t in np.arange(0.3, 0.8, 0.1):
            masks = (probs > t).float()
            mean_ratio = masks.mean().item()
            if 0.05 < mean_ratio < 0.8:  # mantıklı bir aralık
                best_t, best_mean = t, mean_ratio
                break

        return best_t

    def calculate_stats(self, save_csv=True):
        if len(self.dataset) == 0:
            print("🚨 Hesaplama yapılmadı çünkü dataset boş!")
            return [], 0, 0

        start_time = time.time()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        etki_oranlari, sel_hacimleri, results = [], [], []
        total_images, processed_images = len(self.dataset), 0

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(dataloader, 1):
                batch_start = time.time()
                inputs = inputs.to(self.device)

                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                masks = (probs > self.threshold).float().cpu().numpy()

                batch_time = time.time() - batch_start

                for i in range(masks.shape[0]):
                    mask = masks[i].squeeze()
                    if mask.sum() <= 0:
                        processed_images += 1
                        continue

                    oran = (mask.sum() / mask.size) * 100

                    # Uç değerleri kırp (5% - 95%)
                    oran = min(max(oran, 5), 95)

                    hacim = mask.sum() * self.piksel_alani * self.su_yuksekligi
                    etki_oranlari.append(oran)
                    sel_hacimleri.append(hacim)

                    results.append({
                        "Ornek_Index": processed_images,
                        "Sel_Etki_Orani(%)": round(oran, 2),
                        "Sel_Hacmi(m3)": round(hacim, 2)
                    })

                    print(f"🖼️ Fotoğraf {processed_images}/{total_images} işlendi | Sel Oranı: {oran:.2f}%")
                    processed_images += 1

                print(f"📦 Batch {batch_idx} tamamlandı | Süre: {batch_time:.2f} sn")

        avg_oran = np.mean(etki_oranlari) if etki_oranlari else 0
        avg_hacim = np.mean(sel_hacimleri) if sel_hacimleri else 0
        total_time = time.time() - start_time

        print("\n===== SONUÇLAR =====")
        print(f"📊 Ortalama Sel Etki Oranı: {avg_oran:.2f}%")
        print(f"📊 Ortalama Sel Büyüklüğü: {avg_hacim:,.0f} m³")
        print(f"⏱️ Toplam Süre: {total_time:.2f} saniye ({total_time / 60:.2f} dakika)")

        if save_csv:
            csv_path = os.path.join(self.base_dir, "sel_istatistikleri_final.csv")
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Ornek_Index", "Sel_Etki_Orani(%)", "Sel_Hacmi(m3)"])
                writer.writeheader()
                writer.writerows(results)

            print(f"💾 Sonuçlar CSV olarak kaydedildi: {csv_path}")

        return results, avg_oran, avg_hacim


if __name__ == "__main__":
    print("🚀 FloodStats çalışıyor... (Otomatik threshold seçimi + uç değer düzeltme)")
    stats = FloodStats(
        model_path="KaydetmeNoktasi/en_iyi_model.pth",
        base_dir=".",
        threshold=None,  # threshold verilmezse otomatik bulunur
        batch_size=4
    )
    stats.calculate_stats(save_csv=True)
