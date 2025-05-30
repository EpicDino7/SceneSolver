import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PretrainedViTEvidenceModel:
    """Improved Pretrained ViT model for evidence classification with better calibration"""
    
    def __init__(self, model_path=None, model_name="google/vit-base-patch16-224"):
        self.device = device
        self.model_name = model_name
        
        # Evidence classes
        self.evidence_classes = [
            "Gun", "Knife", "Mask", "Car", "Fire", "Glass", "Crowd",
            "Blood", "Explosion", "Bag", "Money", "Weapon", "Smoke", "Person"
        ]
        
        # Improved calibration parameters
        self.base_threshold = 0.55  # Higher base threshold
        self.confidence_gap = 0.05  # Minimum gap between top predictions
        self.max_evidences = 6     # Maximum number of evidences to return
        
        # Load pretrained ViT model
        self.model = ViTModel.from_pretrained(model_name).to(device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        # Evidence classifier (matches training structure)
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
            nn.Linear(256, len(self.evidence_classes))
        ).to(device)
        
        # Load fine-tuned weights if available
        if model_path and self._model_exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️ No fine-tuned ViT model found, using pretrained weights only")
        
        self.model.eval()
        self.evidence_classifier.eval()
    
    def _model_exists(self, model_path):
        """Check if model file exists"""
        import os
        return os.path.exists(model_path)
    
    def _load_model(self, model_path):
        """Load fine-tuned model weights"""
        print(f"🔍 Loading pretrained ViT evidence model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.evidence_classifier.load_state_dict(checkpoint['evidence_classifier_state_dict'])
            
            # Display model metadata
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"✅ Pretrained ViT evidence model loaded successfully")
                print(f"   Training mode: {config.get('training_mode', 'unknown')}")
                print(f"   Images per class: {config.get('images_per_class', 'unknown')}")
                print(f"   Total training images: {config.get('total_images', 'unknown')}")
                print(f"   Best accuracy: {checkpoint.get('best_acc', 'unknown'):.2f}%")
                
                if 'evidence_accuracies' in checkpoint:
                    print("   Per-evidence accuracies:")
                    for evidence, acc in checkpoint['evidence_accuracies'].items():
                        print(f"     {evidence}: {acc:.1f}%")
            else:
                print("✅ Pretrained ViT evidence model loaded successfully")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Using pretrained weights only")
    
    def _apply_adaptive_threshold(self, probabilities, base_threshold=None):
        """
        Apply adaptive thresholding based on score distribution
        """
        if base_threshold is None:
            base_threshold = self.base_threshold
            
        # Convert to numpy for easier manipulation
        probs = probabilities.cpu().numpy() if torch.is_tensor(probabilities) else probabilities
        
        # Calculate statistics
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)
        max_prob = np.max(probs)
        
        # Adaptive threshold calculation
        # If all probabilities are clustered around 0.5, use higher threshold
        if std_prob < 0.01 and mean_prob < 0.51:
            adaptive_threshold = max(base_threshold, mean_prob + 3 * std_prob)
        else:
            # Use statistical threshold: mean + 1.5 * std
            adaptive_threshold = max(base_threshold, mean_prob + 1.5 * std_prob)
        
        # Ensure we don't set threshold too high
        adaptive_threshold = min(adaptive_threshold, 0.85)
        
        return adaptive_threshold
    
    def _filter_top_evidences(self, evidence_results):
        """
        Filter evidences based on confidence gaps and maximum count
        """
        if len(evidence_results) <= 1:
            return evidence_results
            
        # Sort by confidence
        sorted_evidence = sorted(evidence_results, key=lambda x: x["confidence"], reverse=True)
        
        # Apply confidence gap filtering
        filtered_evidence = [sorted_evidence[0]]  # Always include the top prediction
        
        for i in range(1, len(sorted_evidence)):
            current_conf = sorted_evidence[i]["confidence"]
            prev_conf = filtered_evidence[-1]["confidence"]
            
            # Only include if confidence gap is sufficient
            if (prev_conf - current_conf) <= self.confidence_gap and len(filtered_evidence) < self.max_evidences:
                filtered_evidence.append(sorted_evidence[i])
            elif len(filtered_evidence) >= self.max_evidences:
                break
                
        return filtered_evidence
    
    def extract_evidence(self, image, threshold=None, use_adaptive=True):
        """
        Extract evidence from image using improved calibration
        
        Args:
            image: PIL Image or path to image
            threshold: Manual threshold (if None, uses adaptive)
            use_adaptive: Whether to use adaptive thresholding
            
        Returns:
            List of detected evidence with confidence scores
        """
        # Handle image input
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            # Convert numpy array or tensor to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Preprocess image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        with torch.no_grad():
            # Forward pass through ViT
            outputs = self.model(**inputs)
            pooled_output = outputs.pooler_output  # [CLS] token representation
            
            # Evidence classification
            evidence_logits = self.evidence_classifier(pooled_output)
            evidence_probs = torch.sigmoid(evidence_logits).squeeze(0)
        
        # Determine threshold
        if threshold is None and use_adaptive:
            threshold = self._apply_adaptive_threshold(evidence_probs)
            print(f"🎯 Using adaptive threshold: {threshold:.3f}")
        elif threshold is None:
            threshold = self.base_threshold
        
        # Extract evidence above threshold
        evidence_results = []
        for idx, prob in enumerate(evidence_probs):
            if prob.item() > threshold:
                evidence_results.append({
                    "evidence": self.evidence_classes[idx],
                    "confidence": prob.item()
                })
        
        # Apply filtering for better results
        if len(evidence_results) > 3:  # Only filter if we have many predictions
            evidence_results = self._filter_top_evidences(evidence_results)
        
        # Sort by confidence
        evidence_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return evidence_results
    
    def extract_evidence_batch(self, images, threshold=None, use_adaptive=True):
        """
        Extract evidence from batch of images with improved calibration
        
        Args:
            images: List of PIL Images
            threshold: Manual threshold (if None, uses adaptive per image)
            use_adaptive: Whether to use adaptive thresholding
            
        Returns:
            List of evidence results for each image
        """
        # Preprocess batch
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        with torch.no_grad():
            # Forward pass through ViT
            outputs = self.model(**inputs)
            pooled_output = outputs.pooler_output  # [CLS] token representations
            
            # Evidence classification
            evidence_logits = self.evidence_classifier(pooled_output)
            evidence_probs = torch.sigmoid(evidence_logits)
        
        # Extract evidence for each image
        batch_results = []
        for i in range(evidence_probs.size(0)):
            image_probs = evidence_probs[i]
            
            # Determine threshold for this image
            if threshold is None and use_adaptive:
                img_threshold = self._apply_adaptive_threshold(image_probs)
            elif threshold is None:
                img_threshold = self.base_threshold
            else:
                img_threshold = threshold
            
            # Extract evidence
            image_results = []
            for idx, prob in enumerate(image_probs):
                if prob.item() > img_threshold:
                    image_results.append({
                        "evidence": self.evidence_classes[idx],
                        "confidence": prob.item()
                    })
            
            # Apply filtering for better results
            if len(image_results) > 3:
                image_results = self._filter_top_evidences(image_results)
            
            # Sort by confidence
            image_results.sort(key=lambda x: x["confidence"], reverse=True)
            batch_results.append(image_results)
        
        return batch_results
    
    def get_all_evidence_scores(self, image):
        """
        Get confidence scores for all evidence classes
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Dictionary mapping evidence class to confidence score
        """
        # Handle image input
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Preprocess image
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        with torch.no_grad():
            # Forward pass through ViT
            outputs = self.model(**inputs)
            pooled_output = outputs.pooler_output
            
            # Evidence classification
            evidence_logits = self.evidence_classifier(pooled_output)
            evidence_probs = torch.sigmoid(evidence_logits).squeeze(0)
        
        # Create evidence scores dictionary
        evidence_scores = {}
        for idx, prob in enumerate(evidence_probs):
            evidence_scores[self.evidence_classes[idx]] = prob.item()
        
        return evidence_scores
    
    def extract_evidence_with_stats(self, image):
        """
        Extract evidence with detailed statistics for analysis
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Dictionary with evidence results and statistics
        """
        scores = self.get_all_evidence_scores(image)
        score_values = list(scores.values())
        
        # Calculate adaptive threshold
        adaptive_threshold = self._apply_adaptive_threshold(np.array(score_values))
        
        # Extract evidence with adaptive threshold
        evidence_results = self.extract_evidence(image, threshold=adaptive_threshold, use_adaptive=False)
        
        # Create detailed response
        result = {
            "evidence": evidence_results,
            "statistics": {
                "mean_confidence": np.mean(score_values),
                "max_confidence": np.max(score_values),
                "min_confidence": np.min(score_values),
                "std_confidence": np.std(score_values),
                "adaptive_threshold": adaptive_threshold,
                "total_detections": len(evidence_results)
            },
            "all_scores": scores
        }
        
        return result

# Global model instance
vit_evidence_model = None

def load_vit_evidence_model(model_path="vit_pretrained_evidence.pth"):
    """Load the global ViT evidence model"""
    global vit_evidence_model
    vit_evidence_model = PretrainedViTEvidenceModel(model_path)
    return vit_evidence_model

def extract_evidence_from_image(image, threshold=None, use_adaptive=True):
    """
    Convenience function to extract evidence from image with improved calibration
    
    Args:
        image: PIL Image or path to image
        threshold: Manual threshold (if None, uses adaptive)
        use_adaptive: Whether to use adaptive thresholding
        
    Returns:
        List of detected evidence with confidence scores
    """
    global vit_evidence_model
    
    if vit_evidence_model is None:
        vit_evidence_model = load_vit_evidence_model()
    
    return vit_evidence_model.extract_evidence(image, threshold, use_adaptive)

# Legacy function for compatibility with existing code
def extract_evidence(vit_model, image, transform, evidence_classes, threshold=0.3):
    """
    Legacy function for compatibility with existing ViT evidence extraction
    This wraps the new pretrained ViT model to maintain API compatibility
    """
    # Use the global pretrained model if available
    global vit_evidence_model
    
    if vit_evidence_model is None:
        vit_evidence_model = load_vit_evidence_model()
    
    # Extract evidence using the new model with adaptive thresholding
    evidence_results = vit_evidence_model.extract_evidence(image, threshold=None, use_adaptive=True)
    
    # Convert to legacy format (list of evidence names)
    evidence_list = [result["evidence"] for result in evidence_results]
    
    return evidence_list 