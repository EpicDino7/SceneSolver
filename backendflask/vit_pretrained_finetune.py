import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm
import json
import random
import shutil
import time

class FewShotEvidenceDataset(Dataset):
    def __init__(self, image_folder, feature_extractor, crime_classes, images_per_class=300, use_augmentation=True):
        self.image_folder = image_folder
        self.feature_extractor = feature_extractor
        self.crime_classes = crime_classes
        self.images_per_class = images_per_class
        self.use_augmentation = use_augmentation
        
        # Evidence classes for multi-label classification
        self.evidence_classes = [
            "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
            "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
        ]
        
        # Crime-to-evidence mapping
        self.crime_to_evidence = {
            "Arrest": ["Person", "Crowd"],
            "Arson": ["Fire", "Smoke"],
            "Assault": ["Blood", "Person"],
            "Burglary": ["Bag", "Glass"],
            "Explosion": ["Explosion", "Smoke"],
            "Fighting": ["Crowd", "Blood"],
            "RoadAccidents": ["Car", "Person"],
            "Robbery": ["Gun", "Bag", "Mask"],
            "Shooting": ["Gun", "Crowd"],
            "Shoplifting": ["Bag", "Money"],
            "Stealing": ["Bag", "Person"],
            "Vandalism": ["Glass", "Crowd"],
            "Abuse": ["Person"],
            "NormalVideos": []
        }
        
        self.image_paths = []
        self.labels = []
        self.evidence_labels = []
        
        print(f"🎯 Sampling {images_per_class} images per class for ViT evidence classification...")
        
        for class_idx, class_name in enumerate(crime_classes):
            class_dir = os.path.join(image_folder, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Class directory not found: {class_name}")
                continue
                
            # Get all valid image files
            all_images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Sample images for few-shot learning
            sampled_count = min(len(all_images), images_per_class)
            sampled_images = random.sample(all_images, sampled_count)
            
            print(f"  📸 {class_name:15}: {sampled_count:3d}/{len(all_images):5d} images sampled")
            
            for img_file in sampled_images:
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.labels.append(class_idx)
                # Generate evidence labels based on crime type
                evidence_vector = self.encode_evidence_labels(class_name)
                self.evidence_labels.append(evidence_vector)
        
        print(f"✅ Total ViT evidence dataset: {len(self.image_paths)} images")
        
        # Data augmentation for training
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(degrees=20),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
        ]) if use_augmentation else None
        
    def encode_evidence_labels(self, crime_class):
        """Convert crime class to multi-label evidence vector"""
        evidences = self.crime_to_evidence.get(crime_class, [])
        binary_vector = [1 if evidence in evidences else 0 for evidence in self.evidence_classes]
        return torch.tensor(binary_vector, dtype=torch.float32)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        evidence_label = self.evidence_labels[idx]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply augmentation if enabled
            if self.use_augmentation and self.augmentation:
                image = self.augmentation(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        return image, label, evidence_label

class PretrainedViTFineTuner:
    """Fine-tuner for pretrained ViT model on evidence classification"""
    
    def __init__(self, model_name="google/vit-base-patch16-224", num_evidence_classes=14):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Using device: {self.device}")
        
        # Load pretrained ViT model
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        # Multi-label evidence classifier
        self.evidence_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_evidence_classes)
        ).to(self.device)
        
        # Freeze early layers for efficient training
        self.freeze_early_layers(freeze_ratio=0.7)
        
    def freeze_early_layers(self, freeze_ratio=0.7):
        """Freeze early transformer layers for efficient fine-tuning"""
        encoder_layers = list(self.model.encoder.layer)
        freeze_count = int(len(encoder_layers) * freeze_ratio)
        
        # Freeze embeddings
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze early encoder layers
        for i in range(freeze_count):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
        
        print(f"🧊 Frozen embeddings and {freeze_count}/{len(encoder_layers)} encoder layers")
        print(f"🎯 Training {len(encoder_layers)-freeze_count} encoder layers + evidence classifier")
    
    def custom_collate_fn(self, batch):
        """Custom collate function for batching"""
        images, labels, evidence_labels = zip(*batch)
        
        # Process images with feature extractor (keep on CPU)
        inputs = self.feature_extractor(
            images=list(images), 
            return_tensors="pt"
        )
        
        # Keep tensors on CPU for pin memory to work
        labels = torch.tensor(labels, dtype=torch.long)
        evidence_labels = torch.stack(evidence_labels)
        
        return inputs, labels, evidence_labels
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        self.evidence_classifier.train()
        
        total_loss = 0
        total_samples = 0
        total_correct_evidence = 0
        
        loop = tqdm(dataloader, desc=f"ViT Training Epoch {epoch+1}")
        
        for batch_idx, (inputs, labels, evidence_labels) in enumerate(loop):
            optimizer.zero_grad()
            
            # Move inputs to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            labels = labels.to(self.device)
            evidence_labels = evidence_labels.to(self.device)
            
            # Forward pass through ViT
            outputs = self.model(**inputs)
            pooled_output = outputs.pooler_output  # [CLS] token representation
            
            # Evidence classification
            evidence_logits = self.evidence_classifier(pooled_output)
            
            # Multi-label BCE loss for evidence classification
            loss = criterion(evidence_logits, evidence_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.evidence_classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            total_samples += labels.size(0)
            
            # Calculate evidence accuracy (threshold = 0.5)
            evidence_preds = (torch.sigmoid(evidence_logits) > 0.5).float()
            evidence_correct = (evidence_preds == evidence_labels).float().mean()
            total_correct_evidence += evidence_correct.item()
            
            avg_evidence_acc = 100.0 * total_correct_evidence / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(
                loss=loss.item(), 
                evidence_acc=f"{avg_evidence_acc:.2f}%", 
                lr=f"{current_lr:.2e}"
            )
        
        avg_loss = total_loss / len(dataloader)
        avg_evidence_acc = 100.0 * total_correct_evidence / len(dataloader)
        
        return avg_loss, avg_evidence_acc
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        self.evidence_classifier.eval()
        
        total_loss = 0
        total_samples = 0
        total_correct_evidence = 0
        per_class_correct = torch.zeros(14)
        per_class_total = torch.zeros(14)
        confidence_scores = []
        
        evidence_classes = [
            "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
            "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
        ]
        
        with torch.no_grad():
            for inputs, labels, evidence_labels in tqdm(dataloader, desc="Validating", leave=False):
                # Move inputs to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                labels = labels.to(self.device)
                evidence_labels = evidence_labels.to(self.device)
                
                outputs = self.model(**inputs)
                pooled_output = outputs.pooler_output
                
                evidence_logits = self.evidence_classifier(pooled_output)
                loss = criterion(evidence_logits, evidence_labels)
                total_loss += loss.item()
                
                # Evidence predictions
                evidence_probs = torch.sigmoid(evidence_logits)
                evidence_preds = (evidence_probs > 0.5).float()
                
                # Overall evidence accuracy
                evidence_correct = (evidence_preds == evidence_labels).float().mean()
                total_correct_evidence += evidence_correct.item()
                
                # Per-class evidence accuracy
                for i in range(evidence_labels.size(1)):  # For each evidence class
                    class_correct = (evidence_preds[:, i] == evidence_labels[:, i]).sum().item()
                    per_class_correct[i] += class_correct
                    per_class_total[i] += evidence_labels.size(0)
                
                # Confidence scores
                max_probs = evidence_probs.max(dim=1)[0]
                confidence_scores.extend(max_probs.cpu().tolist())
                
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        avg_evidence_acc = 100.0 * total_correct_evidence / len(dataloader)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Per-class evidence accuracies
        evidence_accuracies = {}
        for i, evidence_name in enumerate(evidence_classes):
            if per_class_total[i] > 0:
                evidence_acc = 100.0 * per_class_correct[i] / per_class_total[i]
                evidence_accuracies[evidence_name] = evidence_acc
        
        return avg_loss, avg_evidence_acc, evidence_accuracies, avg_confidence
    
    def save_model(self, path, epoch, best_acc, evidence_accuracies=None, avg_confidence=None, training_info=None):
        """Save model with training metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'evidence_classifier_state_dict': self.evidence_classifier.state_dict(),
            'best_acc': best_acc,
            'avg_confidence': avg_confidence,
            'evidence_accuracies': evidence_accuracies or {},
            'training_info': training_info or {},
            'model_config': {
                'freeze_ratio': 0.7,
                'training_mode': 'pretrained_vit_evidence',
                'images_per_class': training_info.get('images_per_class', 'unknown'),
                'total_images': training_info.get('total_images', 'unknown'),
                'model_type': 'pretrained_vit_evidence',
                'evidence_classes': [
                    "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
                    "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
                ]
            }
        }
        torch.save(checkpoint, path)
        print(f"💾 Pretrained ViT evidence model saved to {path}")

def create_balanced_subset(base_dir, output_dir, images_per_class=300):
    """Create balanced subset for few-shot learning"""
    crime_classes = [
        "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", 
        "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", 
        "Vandalism", "Abuse", "NormalVideos"
    ]
    
    print(f"📁 Creating balanced ViT subset: {images_per_class} images per class")
    print(f"📂 Source: {base_dir}")
    print(f"📂 Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    total_copied = 0
    
    for class_name in crime_classes:
        source_class_dir = os.path.join(base_dir, class_name)
        target_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"⚠️ Skipping {class_name} - source directory not found")
            continue
            
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Get all images
        all_images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Sample images
        sample_count = min(len(all_images), images_per_class)
        sampled_images = random.sample(all_images, sample_count)
        
        # Copy sampled images
        for img_file in sampled_images:
            source_path = os.path.join(source_class_dir, img_file)
            target_path = os.path.join(target_class_dir, img_file)
            shutil.copy2(source_path, target_path)
        
        total_copied += sample_count
        print(f"  ✅ {class_name:15}: {sample_count:3d}/{len(all_images):5d} images copied")
    
    print(f"🎯 Balanced ViT subset created: {total_copied} total images")
    return output_dir

def main():
    """Main training function"""
    CONFIG = {
        'train_dir': r"C:/Users/adity/Downloads/ucf_dataset/Train",
        'images_per_class': 300,  # Few-shot learning
        'batch_size': 16,  # Smaller batch size for ViT
        'learning_rate': 2e-5,  # Lower learning rate for pretrained model
        'epochs': 10,
        'num_evidence_classes': 14,
        'save_path': 'vit_pretrained_evidence.pth',
        'weight_decay': 0.01,
        'create_subset': False  # Set to True to create balanced subset
    }
    
    crime_classes = [
        "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", 
        "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", 
        "Vandalism", "Abuse", "NormalVideos"
    ]
    
    print("🎯 PRETRAINED ViT EVIDENCE CLASSIFICATION TRAINING")
    print("="*70)
    print(f"🤖 Using pretrained google/vit-base-patch16-224")
    print(f"📸 Using only {CONFIG['images_per_class']} images per class")
    print(f"🎯 Training for evidence detection in crime scenes")
    print(f"🚀 Total training images: ~{CONFIG['images_per_class'] * len(crime_classes):,}")
    print(f"📊 Configuration: {CONFIG}")
    
    # Create subset if needed
    if CONFIG['create_subset']:
        subset_dir = f"ucf_vit_subset_{CONFIG['images_per_class']}_per_class"
        CONFIG['train_dir'] = create_balanced_subset(
            CONFIG['train_dir'], 
            subset_dir, 
            CONFIG['images_per_class']
        )
    
    # Initialize model
    fine_tuner = PretrainedViTFineTuner(num_evidence_classes=CONFIG['num_evidence_classes'])
    
    # Create dataset
    full_dataset = FewShotEvidenceDataset(
        CONFIG['train_dir'], 
        fine_tuner.feature_extractor, 
        crime_classes,
        images_per_class=CONFIG['images_per_class'],
        use_augmentation=True
    )
    
    # Train/validation split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset.dataset.use_augmentation = False
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        collate_fn=fine_tuner.custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        collate_fn=fine_tuner.custom_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 Dataset split: {train_size:,} train, {val_size:,} validation")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW([
        {'params': fine_tuner.model.parameters(), 'lr': CONFIG['learning_rate']},
        {'params': fine_tuner.evidence_classifier.parameters(), 'lr': CONFIG['learning_rate'] * 3}
    ], weight_decay=CONFIG['weight_decay'])
    
    # Multi-label BCE loss for evidence classification
    criterion = nn.BCEWithLogitsLoss()
    
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=CONFIG['learning_rate'] * 0.01
    )
    
    # Training loop
    best_acc = 0.0
    patience = 4
    patience_counter = 0
    
    training_info = {
        'images_per_class': CONFIG['images_per_class'],
        'total_images': len(full_dataset),
        'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Pretrained ViT Evidence Training")
        print(f"{'='*70}")
        
        train_loss, train_acc = fine_tuner.train_epoch(
            train_loader, optimizer, criterion, epoch, scheduler
        )
        
        val_loss, val_acc, evidence_accuracies, avg_confidence = fine_tuner.validate(
            val_loader, criterion
        )
        
        print(f"📈 Train Loss: {train_loss:.4f}, Train Evidence Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}, Val Evidence Acc: {val_acc:.2f}%")
        print(f"🎯 Avg Confidence: {avg_confidence:.3f}")
        
        print("🔍 Per-evidence-class validation accuracies:")
        for evidence_name, acc in evidence_accuracies.items():
            emoji = "✅" if acc > 70 else "⚠️" if acc > 50 else "❌"
            print(f"  {emoji} {evidence_name:12}: {acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            training_info['training_time'] = time.time() - start_time
            fine_tuner.save_model(
                CONFIG['save_path'], epoch, best_acc, 
                evidence_accuracies, avg_confidence, training_info
            )
            print(f"🎯 New best validation evidence accuracy: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"⏰ Early stopping: no improvement for {patience} epochs")
            break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
    
    total_time = time.time() - start_time
    
    print(f"\n🏆 Pretrained ViT evidence training completed in {total_time/60:.1f} minutes!")
    print(f"🎯 Best validation evidence accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved to: {CONFIG['save_path']}")
    print(f"📸 Trained on only {CONFIG['images_per_class']} images per class")
    print(f"⚡ Total images used: {len(full_dataset):,}")
    print(f"🔍 Model specialized for evidence detection in crime scenes")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    main() 