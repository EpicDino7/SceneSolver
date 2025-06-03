import torch
import os
from PIL import Image
import numpy as np
from vit_pretrained_model import PretrainedViTEvidenceModel, load_vit_evidence_model

def test_vit_model():
    """Test the pretrained ViT evidence model"""
    print("🧪 Testing Pretrained ViT Evidence Model")
    print("="*50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    try:
        # Initialize model (without fine-tuned weights for testing)
        print("\n🔧 Initializing pretrained ViT model...")
        model = PretrainedViTEvidenceModel(model_path=None)
        print("✅ Model initialized successfully")
        
        # Create a test image
        print("\n📸 Creating test image...")
        test_image = Image.new('RGB', (224, 224), (128, 128, 128))
        print("✅ Test image created")
        
        # Test evidence extraction
        print("\n🔍 Testing evidence extraction...")
        evidence_results = model.extract_evidence(test_image, threshold=0.1)
        print(f"✅ Evidence extraction completed")
        print(f"   Found {len(evidence_results)} pieces of evidence")
        
        for result in evidence_results[:5]:  # Show top 5
            print(f"   - {result['evidence']}: {result['confidence']:.3f}")
        
        # Test batch processing
        print("\n📦 Testing batch evidence extraction...")
        test_images = [test_image] * 3
        batch_results = model.extract_evidence_batch(test_images, threshold=0.1)
        print(f"✅ Batch processing completed")
        print(f"   Processed {len(batch_results)} images")
        
        # Test all evidence scores
        print("\n📊 Testing all evidence scores...")
        all_scores = model.get_all_evidence_scores(test_image)
        print(f"✅ All evidence scores computed")
        print("   Top evidence scores:")
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for evidence, score in sorted_scores[:5]:
            print(f"   - {evidence}: {score:.3f}")
        
        print(f"\n🎉 All tests passed successfully!")
        print(f"📋 Model Info:")
        print(f"   - Evidence classes: {len(model.evidence_classes)}")
        print(f"   - Model device: {model.device}")
        print(f"   - Feature extractor: {model.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_global_model_loading():
    """Test the global model loading functionality"""
    print("\n🌐 Testing global model loading...")
    
    try:
        # Test loading without existing model file
        model = load_vit_evidence_model("nonexistent_model.pth")
        print("✅ Global model loaded (with warnings expected)")
        
        # Test convenience function
        test_image = Image.new('RGB', (224, 224), (64, 64, 64))
        from vit_pretrained_model import extract_evidence_from_image
        
        results = extract_evidence_from_image(test_image, threshold=0.1)
        print(f"✅ Convenience function works: {len(results)} evidence found")
        
        return True
        
    except Exception as e:
        print(f"❌ Global model test failed: {e}")
        return False

def test_model_info():
    """Display model architecture information"""
    print("\n🏗️ Model Architecture Information")
    print("="*50)
    
    try:
        model = PretrainedViTEvidenceModel(model_path=None)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in model.evidence_classifier.parameters())
        
        print(f"📊 Parameter Information:")
        print(f"   - Total ViT parameters: {total_params:,}")
        print(f"   - Trainable ViT parameters: {trainable_params:,}")
        print(f"   - Evidence classifier parameters: {classifier_params:,}")
        print(f"   - Frozen parameters: {total_params - trainable_params:,}")
        print(f"   - Freeze ratio: {((total_params - trainable_params) / total_params) * 100:.1f}%")
        
        print(f"\n🎯 Evidence Classes ({len(model.evidence_classes)}):")
        for i, evidence in enumerate(model.evidence_classes):
            print(f"   {i+1:2d}. {evidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ViT Pretrained Model Tests")
    print("="*60)
    
    # Run all tests
    test_results = []
    
    test_results.append(("Basic Model Test", test_vit_model()))
    test_results.append(("Global Loading Test", test_global_model_loading()))
    test_results.append(("Model Info Test", test_model_info()))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Install dependencies: pip install -r requirements_vit.txt")
        print("2. Run training: python vit_pretrained_finetune.py")
        print("3. Integrate with pipeline: Update crime_pipeline_few_shot.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.") 