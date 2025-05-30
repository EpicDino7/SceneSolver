# Pretrained ViT Evidence Classification Model

This implementation provides a pretrained Vision Transformer (ViT) model fine-tuned for evidence classification in crime scene analysis, similar to the CLIP model approach but specialized for evidence detection.

## 🎯 Overview

The pretrained ViT model is designed to:

- **Extract evidence** from crime scene images
- **Multi-label classification** for 14 evidence types
- **Few-shot learning** with only 300 images per crime class
- **Transfer learning** from Google's pretrained ViT-Base-Patch16-224
- **Efficient training** with frozen early layers

## 📁 Files

### Core Implementation

- `vit_pretrained_finetune.py` - Main training script for fine-tuning ViT
- `vit_pretrained_model.py` - Model loading and inference utilities
- `test_vit_pretrained.py` - Test script to verify implementation
- `requirements_vit.txt` - Required dependencies

### Integration

- Compatible with existing `crime_pipeline_few_shot.py`
- Can replace or complement the current ViT model in `vit_new.py`

## 🔧 Installation

1. **Install dependencies:**

```bash
pip install -r requirements_vit.txt
```

2. **Verify installation:**

```bash
python test_vit_pretrained.py
```

## 🚀 Training

### Quick Start

```bash
python vit_pretrained_finetune.py
```

### Configuration

Edit the `CONFIG` dictionary in `vit_pretrained_finetune.py`:

```python
CONFIG = {
    'train_dir': r"C:/Users/adity/Downloads/ucf_dataset/Train",
    'images_per_class': 300,  # Few-shot learning
    'batch_size': 16,         # Adjust based on GPU memory
    'learning_rate': 2e-5,    # Lower for pretrained model
    'epochs': 10,
    'save_path': 'vit_pretrained_evidence.pth',
    'create_subset': False    # Set True to create balanced subset
}
```

### Training Features

- **Pretrained weights**: Starts from Google's ViT-Base-Patch16-224
- **Layer freezing**: 70% of early layers frozen for efficiency
- **Data augmentation**: Comprehensive augmentation pipeline
- **Multi-label BCE loss**: For evidence classification
- **Early stopping**: Prevents overfitting
- **Progress tracking**: Real-time training metrics

## 📊 Evidence Classes

The model detects 14 types of evidence:

1. Gun
2. Knife
3. Mask
4. Car
5. Fire
6. Glass
7. Crowd
8. Blood
9. Explosion
10. Bag
11. Money
12. Weapon
13. Smoke
14. Person

## 🎮 Usage

### Basic Inference

```python
from vit_pretrained_model import PretrainedViTEvidenceModel
from PIL import Image

# Load model
model = PretrainedViTEvidenceModel("vit_pretrained_evidence.pth")

# Load image
image = Image.open("crime_scene.jpg")

# Extract evidence
evidence = model.extract_evidence(image, threshold=0.4)
print(evidence)
# Output: [{"evidence": "Gun", "confidence": 0.87}, ...]
```

### Batch Processing

```python
# Process multiple images
images = [Image.open(f"frame_{i}.jpg") for i in range(10)]
batch_results = model.extract_evidence_batch(images)
```

### All Evidence Scores

```python
# Get confidence for all evidence types
all_scores = model.get_all_evidence_scores(image)
print(all_scores)
# Output: {"Gun": 0.87, "Knife": 0.12, ...}
```

### Global Model (Convenience)

```python
from vit_pretrained_model import extract_evidence_from_image

# Automatically loads model on first call
evidence = extract_evidence_from_image("image.jpg", threshold=0.4)
```

## 🏗️ Model Architecture

```
Pretrained ViT-Base-Patch16-224
├── Patch Embedding (16x16 patches)
├── Position Embedding
├── Transformer Encoder (12 layers)
│   ├── Frozen Layers (0-8)    # 70% frozen
│   └── Trainable Layers (9-11) # 30% fine-tuned
└── Evidence Classifier
    ├── Dropout(0.3)
    ├── Linear(768 → 512) + ReLU + BatchNorm
    ├── Dropout(0.2)
    ├── Linear(512 → 256) + ReLU + BatchNorm
    ├── Dropout(0.1)
    └── Linear(256 → 14) # Multi-label output
```

## 📈 Training Strategy

### Crime-to-Evidence Mapping

```python
crime_to_evidence = {
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
    "Normal": []
}
```

### Few-Shot Learning

- **300 images per crime class** (vs full dataset)
- **~4,200 total images** for training
- **80/20 train/validation split**
- **Balanced sampling** across all crime types

### Transfer Learning

- **Frozen embeddings** for stability
- **Frozen early layers** for efficiency
- **Fine-tuned late layers** for adaptation
- **Custom classifier** for evidence detection

## 🔄 Integration with Existing Pipeline

### Option 1: Replace Current ViT

Update `crime_pipeline_few_shot.py`:

```python
from vit_pretrained_model import load_vit_evidence_model

# Replace existing ViT loading
vit_model = load_vit_evidence_model("vit_pretrained_evidence.pth")
```

### Option 2: Dual Model Approach

Keep both models for different tasks:

- **CLIP**: Crime classification
- **Pretrained ViT**: Evidence detection

### Option 3: Legacy Compatibility

The implementation includes legacy function wrappers:

```python
# Existing code continues to work
from vit_pretrained_model import extract_evidence
evidence = extract_evidence(model, image, transform, evidence_classes)
```

## 🎯 Performance Optimization

### Memory Efficiency

- **Smaller batch size** (16) for ViT
- **Gradient checkpointing** available
- **Mixed precision** compatible

### Speed Optimization

- **Layer freezing** reduces training time
- **Efficient data loading** with num_workers
- **Early stopping** prevents overtraining

### Model Size

- **Base ViT**: ~86M parameters
- **Trainable**: ~25M parameters (30%)
- **Custom classifier**: ~400K parameters
- **Total model size**: ~350MB

## 📊 Expected Results

### Training Metrics

- **Training time**: 2-3 hours (vs 12+ hours full dataset)
- **Memory usage**: 4-6GB GPU
- **Convergence**: 8-10 epochs
- **Best accuracy**: 75-85% (evidence detection)

### Evidence Detection Performance

- **High precision** for visible evidence (Gun, Car, Fire)
- **Good recall** for common evidence (Person, Crowd)
- **Balanced performance** across evidence types

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch_size to 8 or 4
   - Enable gradient checkpointing
   - Use smaller image resolution

2. **Slow Training**

   - Increase num_workers in DataLoader
   - Use SSD storage for dataset
   - Enable mixed precision training

3. **Poor Convergence**

   - Adjust learning rate (try 1e-5 or 3e-5)
   - Increase training epochs
   - Check data quality and balance

4. **Low Evidence Accuracy**
   - Adjust evidence threshold (0.3-0.5)
   - Review crime-to-evidence mapping
   - Add more specific augmentations

### Model Loading Issues

```python
# Debug model loading
import torch
checkpoint = torch.load("vit_pretrained_evidence.pth", map_location="cpu")
print(checkpoint.keys())  # Check available keys
```

## 🚀 Advanced Usage

### Custom Evidence Classes

Modify evidence classes in both training and inference:

```python
custom_evidence = ["Gun", "Knife", "Blood", "Fire"]  # Subset
# Retrain with modified crime_to_evidence mapping
```

### Different ViT Models

Change the base model:

```python
model = PretrainedViTFineTuner(
    model_name="google/vit-large-patch16-224"  # Larger model
)
```

### Ensemble Methods

Combine multiple models:

```python
vit_results = vit_model.extract_evidence(image)
clip_results = clip_model.classify_crime(image)
# Combine predictions
```

## 📚 References

- **ViT Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **HuggingFace ViT**: https://huggingface.co/google/vit-base-patch16-224
- **Transfer Learning**: Fine-tuning pretrained vision transformers
- **Multi-label Classification**: BCE loss for evidence detection

## 🎉 Conclusion

This pretrained ViT implementation provides:

- ✅ **Efficient training** with few-shot learning
- ✅ **High-quality evidence detection**
- ✅ **Easy integration** with existing pipeline
- ✅ **Comprehensive documentation** and testing
- ✅ **Production-ready** code structure

Start training with just 300 images per class and achieve strong evidence detection performance in crime scene analysis!
