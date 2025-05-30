#!/usr/bin/env python3
"""
Test script to verify app integration with the new pretrained ViT model
"""

import os
import sys
from PIL import Image
import numpy as np

def test_model_loading():
    """Test if all models can be loaded successfully"""
    print("🧪 Testing Model Loading")
    print("="*50)
    
    try:
        from crime_pipeline_few_shot import load_models
        
        # Load models
        load_models(
            vit_checkpoint="vit_model.pth",
            clip_checkpoint="clip_finetuned_few_shot.pth"
        )
        
        print("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_pipeline():
    """Test the prediction pipeline with a dummy image"""
    print("\n🔍 Testing Prediction Pipeline")
    print("="*50)
    
    try:
        from crime_pipeline_few_shot import predict_single_image
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), (128, 128, 128))
        test_image_path = "test_image.jpg"
        test_image.save(test_image_path)
        
        # Test prediction
        result = predict_single_image(test_image_path)
        
        if result:
            print("✅ Prediction pipeline works!")
            print(f"   📊 Result keys: {list(result.keys())}")
            print(f"   🎯 Crime: {result.get('predicted_class', 'N/A')}")
            print(f"   🔍 Evidence count: {len(result.get('extracted_evidence', []))}")
            
            # Show evidence details
            evidence = result.get('extracted_evidence', [])
            if evidence:
                print(f"   🔎 Top evidence:")
                for i, ev in enumerate(evidence[:3]):
                    print(f"      {i+1}. {ev['label']}: {ev['confidence']}")
            
            success = True
        else:
            print("❌ Prediction returned None")
            success = False
        
        # Cleanup
        try:
            os.remove(test_image_path)
        except:
            pass
            
        return success
        
    except Exception as e:
        print(f"❌ Prediction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        try:
            os.remove("test_image.jpg")
        except:
            pass
            
        return False

def test_pretrained_vit_direct():
    """This test was removed - reverted to original ViT model"""
    pass

def test_app_health_check():
    """Test if the app health check recognizes the required model files"""
    print("\n🏥 Testing App Health Check")
    print("="*50)
    
    try:
        # Import app components
        import sys
        sys.path.append('.')
        
        # Check if model files exist
        model_files = [
            "clip_finetuned_few_shot.pth",
            "vit_model.pth", 
            "clip_model_weights.pth"
        ]
        
        missing_files = []
        for model_file in model_files:
            if not os.path.exists(model_file):
                missing_files.append(model_file)
        
        if not missing_files:
            print("✅ All model files found!")
            print(f"   📁 Checked files: {model_files}")
        else:
            print(f"⚠️ Missing files: {missing_files}")
            print(f"✅ Found files: {[f for f in model_files if f not in missing_files]}")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting App Integration Tests")
    print("="*60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction_pipeline), 
        ("App Health Check", test_app_health_check)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Integration Test Summary")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests passed! Your app is ready to use the original ViT model.")
        print("\n📚 Next Steps:")
        print("1. Start your Flask app: python app.py")
        print("2. Test with real images via the /upload endpoint")
        print("3. The original scratch ViT model is now active")
    else:
        print("⚠️ Some integration tests failed. Please check the errors above.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 