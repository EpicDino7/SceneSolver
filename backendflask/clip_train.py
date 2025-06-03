import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from clip_model import CrimeClassifier  # your CLIP ViT from scratch
from tqdm import tqdm

def main():
    # ✅ Setup
    IMG_SIZE = 64
    BATCH_SIZE = 128  # reduced for 6GB VRAM
    EPOCHS = 1
    LR = 1e-4
    NUM_CLASSES = 14

    # Local Paths
    TRAIN_DIR = "ucf_dataset\\Train"   # <-- UPDATE this
    MODEL_LOAD_PATH = None   # old saved weights (optional)
    MODEL_SAVE_PATH = "weights_clip.pth"   # overwrite with best model

    crime_classes = [
        "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
        "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "Normal"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    # ✅ Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # ✅ Dataset & Loader
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    val_size = int(0.1 * len(full_dataset))  # 10% validation
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 🏎️ DataLoader optimized
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"🧠 Loaded {len(full_dataset)} images ({train_size} train / {val_size} val)")

    # ✅ Model
    model = CrimeClassifier(num_classes=NUM_CLASSES).to(device)

    # ✅ Load previous weights if available
    best_val_acc = 0.0
    if MODEL_LOAD_PATH and os.path.exists(MODEL_LOAD_PATH):
        print(f"🔁 Loading previous model weights from {MODEL_LOAD_PATH}")
        state_dict = torch.load(MODEL_LOAD_PATH, map_location=device)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"⚠️ Direct load failed: {e}")
            print("🔁 Fixing keys...")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
    else:
        print("🚀 Starting fresh training...")

    # ✅ Optimizer and Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ✅ Training Loop
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Training {epoch+1}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples
        print(f"✅ Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 🔍 Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader, desc="🔎 Validating", leave=False):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_preds = val_outputs.max(1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"📈 Validation Accuracy: {val_acc:.2f}%")

        # ✅ Save best model
        if val_acc > best_val_acc:
            print(f"✅ Validation improved ({best_val_acc:.2f}% → {val_acc:.2f}%), saving model...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_acc = val_acc
        else:
            print(f"⚠️ Validation did not improve ({val_acc:.2f}%).")

    print("\n🎯 Training finished!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")

# ✅ Very important for Windows/PyTorch
if __name__ == "__main__":
    main()
