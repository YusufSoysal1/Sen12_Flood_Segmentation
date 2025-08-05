import torch
from Dataset import FloodDataset
from DeepLabV3Plus import DeepLabV3Plus
from Train import train_model

if __name__ == "__main__":
    dataset = FloodDataset(
        base_dir=".",               # Ana klasör
        use_s1_labels=True,
        use_s2_labels=False,
        use_s2=True,
        target_size=(512, 512)      # Optik & SAR aynı boyutta
    )

    print(f"Toplam örnek sayısı: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    model = DeepLabV3Plus(in_channels=8, out_channels=1)
    train_model(
        model=model,
        dataset=dataset,
        device=device,
        epochs=100,             # Uzun eğitim
        batch_size=4,
        val_split=0.2,
        save_dir="KaydetmeNoktasi"
    )
