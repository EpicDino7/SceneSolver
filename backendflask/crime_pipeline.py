# import os
# import torch
# from PIL import Image
# from clip_new import CustomCLIP  # Your custom CLIP model
# from vit_new import VisionTransformer  # Your ViT model with extract_evidence()
# from transforms import test_transform  # Your common transform
# from tqdm import tqdm

# # Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load Models
# clip_model = CustomCLIP().to(device)
# vit_model = VisionTransformer().to(device)

# clip_model.eval()
# vit_model.eval()

# def load_images(image_paths):
#     images = []
#     for path in image_paths:
#         image = Image.open(path).convert('RGB')
#         image = test_transform(image)
#         images.append(image)
#     return torch.stack(images)

# def run_pipeline(image_paths):
#     images = load_images(image_paths).to(device)

#     results = []

#     with torch.no_grad():
#         # CLIP Classification
#         clip_preds = clip_model.predict(images)

#         # ViT Evidence Extraction
#         evidences = vit_model.extract_evidence(images)

#         for idx, img_path in enumerate(image_paths):
#             result = {
#                 'image_name': os.path.basename(img_path),
#                 'clip_predicted_class': clip_preds[idx],
#                 'extracted_evidence': evidences[idx]
#             }
#             results.append(result)

#     return results


# if __name__ == "__main__":
#     # Example: Process all images from a folder
#     image_folder = "test_images"  # put your test images folder here
#     image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

#     output = run_pipeline(image_paths)

#     for res in output:
#         print(res)

# import os
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# import json

# from clip_new import ImprovedCNN
# from vit_new import VisionTransformer, extract_evidence
# from transforms import test_transform  # your custom transforms from transforms.py

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Class Names (Same order as your training)
# crime_classes = [
#     "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
#     "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "Normal"
# ]

# evidence_classes = [
#     "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
#     "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
# ]

# # Load Models
# # Load Models
# clip_model = ImprovedCNN(num_classes=len(crime_classes)).to(device)
# vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)

# # Load Checkpoint with weights_only=True to avoid FutureWarning
# try:
#     checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
    
#     # Verify keys before loading
#     if 'clip_model' in checkpoint and 'vit_model' in checkpoint:
#         clip_model.load_state_dict(checkpoint['clip_model'])
#         vit_model.load_state_dict(checkpoint['vit_model'])
#     else:
#         raise KeyError("Checkpoint does not contain required keys: 'clip_model' or 'vit_model'")
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# clip_model.eval()
# vit_model.eval()


# # Inference Function
# def predict(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         img_tensor = test_transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # Crime Classification (CLIP)
#             output1 = clip_model(img_tensor)
#             crime_idx = torch.argmax(output1, dim=1).item()
#             predicted_crime = crime_classes[crime_idx]
#             crime_confidence = torch.max(F.softmax(output1, dim=1)).item()

#             # Evidence Extraction (ViT)
#             extracted_evidence = extract_evidence(vit_model, image, test_transform, evidence_classes)

#         return {
#             "image_name": os.path.basename(image_path),
#             "predicted_class": predicted_crime,
#             "crime_confidence": crime_confidence,
#             "extracted_evidence": extracted_evidence
#         }
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None

# # Inference on Folder
# def run_inference_on_folder(folder_path):
#     results = []

#     images = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

#     for img_name in tqdm(images):
#         img_path = os.path.join(folder_path, img_name)
#         result = predict(img_path)
#         if result is not None:
#             results.append(result)

#     # Save results
#     with open('results.json', 'w') as f:
#         json.dump(results, f, indent=4)

#     print("Inference Complete! Results saved to results.json")

# # Run
# if __name__ == "__main__":
#     test_folder = "C:\\Users\\Tusha\\OneDrive\\Desktop\\SceneSolver\\mini_dataset\\Test"  # Put your test images here
#     run_inference_on_folder(test_folder)


# import os
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# import json

# from clip_new import ImprovedCNN
# from vit_new import VisionTransformer, extract_evidence
# from transforms import test_transform  # your test transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# crime_classes = [
#     "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
#     "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "Normal"
# ]

# evidence_classes = [
#     "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
#     "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
# ]

# # Load Models
# clip_model = ImprovedCNN(num_classes=len(crime_classes)).to(device)
# vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)

# checkpoint = torch.load("model.pth", map_location=device, weights_only=True)

# clip_model.load_state_dict(checkpoint['clip_model'])
# vit_model.load_state_dict(checkpoint['vit_model'])

# clip_model.eval()
# vit_model.eval()


# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     img_tensor = test_transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output1 = clip_model(img_tensor)
#         crime_idx = torch.argmax(output1, dim=1).item()
#         predicted_crime = crime_classes[crime_idx]
#         crime_confidence = torch.max(F.softmax(output1, dim=1)).item()

#         extracted_evidence = extract_evidence(vit_model, image, test_transform, evidence_classes)

#     return {
#         "image_name": os.path.basename(image_path),
#         "predicted_class": predicted_crime,
#         "crime_confidence": crime_confidence,
#         "extracted_evidence": extracted_evidence
#     }


# def run_inference_on_folder(folder_path):
#     results = []

#     images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     print(f"Found {len(images)} images in {folder_path}")

#     for image_name in tqdm(images, desc="Running Inference"):
#         image_path = os.path.join(folder_path, image_name)
#         result = predict(image_path)
#         if result:
#             results.append(result)

#     with open("results.json", "w") as f:
#         json.dump(results, f, indent=4)

#     print("Inference Complete! Results saved to results.json")


# if __name__ == "__main__":
#     test_folder = "mini_dataset\\Test\\Abuse"   # <-- put your test images folder name here (relative to your current directory)
#     run_inference_on_folder(test_folder)

# import os
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# import json

# from clip_new import ImprovedCNN
# from vit_new import VisionTransformer, extract_evidence
# from transforms import test_transform  # your custom transforms from transforms.py

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Class Names (Same order as your training)
# crime_classes = [
#     "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
#     "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "Normal"
# ]

# evidence_classes = [
#     "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
#     "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
# ]

# # Load Models
# clip_model = ImprovedCNN(num_classes=len(crime_classes)).to(device)
# vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)

# # Load Checkpoint
# try:
#     checkpoint = torch.load("model.pth", map_location=device, weights_only=True)

#     if 'clip_model' in checkpoint and 'vit_model' in checkpoint:
#         clip_model.load_state_dict(checkpoint['clip_model'])
#         vit_model.load_state_dict(checkpoint['vit_model'])
#     else:
#         raise KeyError("Checkpoint missing 'clip_model' or 'vit_model'")
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# clip_model.eval()
# vit_model.eval()


# # Prediction Function
# def predict(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         img_tensor = test_transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # Crime Classification
#             output1 = clip_model(img_tensor)
#             crime_idx = torch.argmax(output1, dim=1).item()
#             predicted_crime = crime_classes[crime_idx]
#             crime_confidence = torch.max(F.softmax(output1, dim=1)).item()

#             # Evidence Extraction
#             extracted_evidence = extract_evidence(vit_model, image, test_transform, evidence_classes)

#         return {
#             "image_name": os.path.basename(image_path),
#             "predicted_class": predicted_crime,
#             "crime_confidence": crime_confidence,
#             "extracted_evidence": extracted_evidence
#         }

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None


# # Inference on Folder (Recursive Subfolders)
# def run_inference_on_folder(folder_path):
#     results = []

#     images = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 images.append(os.path.join(root, file))

#     print(f"Found {len(images)} images in {folder_path}")

#     for image_path in tqdm(images, desc="Running Inference"):
#         result = predict(image_path)
#         if result:
#             results.append(result)

#     with open("results.json", "w") as f:
#         json.dump(results, f, indent=4)

#     print("Inference Complete! Results saved to results.json")


# # Main Driver
# if __name__ == "__main__":
#     test_folder = "mini_dataset/Test"  # Change path if needed
#     run_inference_on_folder(test_folder)

# import os
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# import json

# from clip_new import ImprovedCNN
# from vit_new import VisionTransformer
# from transforms import test_transform  # Must resize to 64x64

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Class names
# crime_classes = [
#     "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
#     "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "Normal"
# ]

# evidence_classes = [
#     "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
#     "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
# ]

# # Load models
# clip_model = ImprovedCNN(num_classes=len(crime_classes)).to(device)
# vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)

# # Load checkpoint
# try:
#     checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
#     clip_state_dict = checkpoint["clip_model"]
#     print("[CHECK] pos_embed shape loaded into clip_model:", clip_state_dict['patch_embed.pos_embed'].shape)


#     clip_model.load_state_dict(checkpoint["clip_model"])
#     vit_model.load_state_dict(checkpoint["vit_model"])

# except Exception as e:
#     print(f"[ERROR] Failed to load model checkpoint: {e}")

# clip_model.eval()
# vit_model.eval()


# # Updated evidence extraction
# def get_evidence_predictions(vit_output, threshold=0.3):
#     probs = torch.sigmoid(vit_output).squeeze(0).cpu()
#     return [
#         {"label": evidence_classes[i], "confidence": round(probs[i].item(), 3)}
#         for i in range(len(probs)) if probs[i].item() > threshold
#     ]


# # Prediction for a single image
# def predict(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         img_tensor = test_transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # Crime classification (CLIP)
#             crime_logits = clip_model(img_tensor)
#             crime_probs = F.softmax(crime_logits, dim=1)
#             crime_idx = torch.argmax(crime_probs, dim=1).item()

#             # Evidence detection (ViT)
#             evidence_logits = vit_model(img_tensor)
#             evidence_results = get_evidence_predictions(evidence_logits)

#         return {
#             "image_name": os.path.basename(image_path),
#             "predicted_class": crime_classes[crime_idx],
#             "crime_confidence": round(crime_probs[0][crime_idx].item(), 3),
#             "extracted_evidence": evidence_results
#         }

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None


# # Folder inference
# def run_inference_on_folder(folder_path):
#     results = []
#     image_paths = []

#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_paths.append(os.path.join(root, file))

#     print(f"Found {len(image_paths)} images in {folder_path}")

#     for img_path in tqdm(image_paths, desc="Running Inference"):
#         result = predict(img_path)
#         if result:
#             results.append(result)

#     with open("results.json", "w") as f:
#         json.dump(results, f, indent=4)

#     print("✅ Inference complete — results saved to results.json")


# # Entry point
# if __name__ == "__main__":
#     test_folder = "mini_dataset/Test"  # Change path as needed
#     run_inference_on_folder(test_folder)


# import os
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# import json

# from clip_model import CrimeClassifier
# from vit_new import VisionTransformer
# from transforms import test_transform  # Must include Resize(64, 64)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define class names
# crime_classes = [
#     "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
#     "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "NormalVideos"
# ]

# evidence_classes = [
#     "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
#     "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
# ]

# # Initialize models
# clip_model = CrimeClassifier(num_classes=len(crime_classes)).to(device)
# vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)


# def load_models(clip_checkpoint="clip_model_weights.pth", vit_checkpoint="vit_model.pth"):
#     print(f"🔍 Loading CLIP checkpoint from: {clip_checkpoint}")
#     clip_ckpt = torch.load(clip_checkpoint, map_location=device)
#     clip_model.load_state_dict(clip_ckpt)
    
#     print(f"🔍 Loading ViT checkpoint from: {vit_checkpoint}")
#     vit_ckpt = torch.load(vit_checkpoint, map_location=device)
#     vit_model.load_state_dict(vit_ckpt)

#     clip_model.eval()
#     vit_model.eval()
#     print("✅ Models loaded and ready.")


# # Evidence prediction helper
# def get_evidence_predictions(logits, threshold=0.3):
#     probs = torch.sigmoid(logits).squeeze(0).cpu()
#     return [
#         {"label": evidence_classes[i], "confidence": round(probs[i].item(), 3)}
#         for i in range(len(probs)) if probs[i].item() > threshold
#     ]


# # Predict for a single image
# def predict(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         img_tensor = test_transform(image).unsqueeze(0).to(device)

#         with torch.no_grad():
#             # CLIP for crime class
#             crime_logits = clip_model(img_tensor)
#             crime_probs = F.softmax(crime_logits, dim=1)
#             crime_idx = torch.argmax(crime_probs, dim=1).item()
#             predicted_crime = crime_classes[crime_idx]
#             crime_conf = round(crime_probs[0][crime_idx].item(), 3)

#             # ViT for evidence extraction
#             evidence_logits = vit_model(img_tensor)
#             evidence_found = get_evidence_predictions(evidence_logits)

#         return {
#             "image_name": os.path.basename(image_path),
#             "predicted_class": predicted_crime,
#             "crime_confidence": crime_conf,
#             "extracted_evidence": evidence_found
#         }

#     except Exception as e:
#         print(f"[ERROR] Failed on {image_path}: {e}")
#         return None


# # Predict for folder of images
# def run_inference_on_folder(folder_path, output_json="results.json"):
#     results = []
#     image_paths = []

#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_paths.append(os.path.join(root, file))

#     print(f" Found {len(image_paths)} images in: {folder_path}")

#     for img_path in tqdm(image_paths, desc=" Running Inference"):
#         result = predict(img_path)
#         if result:
#             results.append(result)

#     with open(output_json, "w") as f:
#         json.dump(results, f, indent=4)

#     print(f"📁 Inference complete. Results saved to: {output_json}")


# # Main Entry
# if __name__ == "__main__":
#     load_models("save_model.pth")
#     test_folder = "mini_dataset/Test/Robbery"  # Update this if needed
#     run_inference_on_folder(test_folder)

import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import json
import torchvision.transforms as transforms

from clip_model import CrimeClassifier
from vit_new import VisionTransformer
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # ONLY resize, no augmentations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names
crime_classes = [
    "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism", "Abuse", "NormalVideos"
]

evidence_classes = [
    "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
    "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
]

# Initialize models
clip_model = CrimeClassifier(num_classes=len(crime_classes)).to(device)
vit_model = VisionTransformer(num_classes=len(evidence_classes)).to(device)

def load_models(clip_checkpoint="clip_model_weights.pth", vit_checkpoint="vit_model.pth"):
    print(f"🔍 Loading CLIP checkpoint from: {clip_checkpoint}")
    clip_ckpt = torch.load(clip_checkpoint, map_location=device)
    clip_model.load_state_dict(clip_ckpt)
    
    print(f"🔍 Loading ViT checkpoint from: {vit_checkpoint}")
    vit_ckpt = torch.load(vit_checkpoint, map_location=device)
    vit_model.load_state_dict(vit_ckpt)

    clip_model.eval()
    vit_model.eval()
    print("✅ Models loaded and ready.")

# Evidence prediction helper
def get_evidence_predictions(logits, threshold=0.3):
    probs = torch.sigmoid(logits).squeeze(0).cpu()
    return [
        {"label": evidence_classes[i], "confidence": round(probs[i].item(), 3)}
        for i in range(len(probs)) if probs[i].item() > threshold
    ]

# Predict for a single image
def predict(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            crime_logits = clip_model(img_tensor)
            crime_probs = F.softmax(crime_logits, dim=1)
            crime_idx = torch.argmax(crime_probs, dim=1).item()
            predicted_crime = crime_classes[crime_idx]
            crime_conf = round(crime_probs[0][crime_idx].item(), 3)

            evidence_logits = vit_model(img_tensor)
            evidence_found = get_evidence_predictions(evidence_logits)

        return {
            "image_name": os.path.basename(image_path),
            "predicted_class": predicted_crime,
            "crime_confidence": crime_conf,
            "extracted_evidence": evidence_found
        }

    except Exception as e:
        print(f"[ERROR] Failed on {image_path}: {e}")
        return None

# Predict for a set of images and summarize
def predict_multiple(images):
    crime_votes = {}
    all_evidence = []

    for img_path in images:
        result = predict(img_path)
        if not result:
            continue

        # Tally crime predictions
        crime = result["predicted_class"]
        crime_votes[crime] = crime_votes.get(crime, 0) + 1

        # Collect all evidence
        all_evidence.extend(result["extracted_evidence"])

    if not crime_votes:
        return {"error": "No valid predictions"}

    # Determine majority crime class
    final_crime = max(crime_votes.items(), key=lambda x: x[1])[0]
    total_votes = sum(crime_votes.values())
    final_conf = round(crime_votes[final_crime] / total_votes, 3)

    # Aggregate evidence
    evidence_dict = {}
    for e in all_evidence:
        label = e["label"]
        confidence = e["confidence"]
        if label in evidence_dict:
            evidence_dict[label].append(confidence)
        else:
            evidence_dict[label] = [confidence]

    aggregated_evidence = [
        {"label": label, "confidence": round(sum(confs)/len(confs), 3)}
        for label, confs in evidence_dict.items()
        if sum(confs)/len(confs) > 0.3  # Apply threshold again after averaging
    ]

    return {
        "predicted_class": final_crime,
        "crime_confidence": final_conf,
        "extracted_evidence": aggregated_evidence
    }

# Main Entry (for optional manual testing)
if __name__ == "__main__":
    load_models(clip_checkpoint="clip_model_weights.pth", vit_checkpoint="vit_model.pth")
    # Optional manual test
    # test_folder = "some/folder/of/images"
    # run_inference_on_folder(test_folder)
