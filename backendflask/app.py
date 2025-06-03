from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import os
import json
import shutil
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import the crime pipeline, handle case where models are missing
try:
    from crime_pipeline_few_shot import load_models, predict_single_image, predict_multiple_images, process_crime_scene, extract_frames
    MODELS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not load models - {e}")
    print("💡 Run 'python download_models.py' to set up model files")
    MODELS_AVAILABLE = False

app = Flask(__name__)
CORS(app) 

# Constants - now using environment variables with defaults
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
FRAMES_FOLDER = os.getenv('FRAMES_FOLDER', 'frames')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models if available
if MODELS_AVAILABLE:
    try:
        load_models(
            vit_checkpoint="vit_model.pth",
            clip_checkpoint="clip_finetuned_few_shot.pth"
        )
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("💡 Run 'python download_models.py' to set up model files")
        MODELS_AVAILABLE = False

# MongoDB setup - now using environment variables
try:
    mongodb_url = os.getenv('MONGODB_URL')
    if not mongodb_url:
        raise ValueError("MONGODB_URL environment variable not set")
    
    client = MongoClient(mongodb_url)
    db_name = os.getenv('MONGODB_DB_NAME', 'crime_db')
    collection_name = os.getenv('MONGODB_COLLECTION_NAME', 'predictions')
    
    db = client[db_name]
    collection = db[collection_name]
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    collection = None

def cleanup_files(case_dir, frames_dir=None):
    """Clean up local files after analysis."""
    try:
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}

def process_uploaded_files(files, case_dir):
    """Process uploaded files, handling both images and videos."""
    image_paths = []
    frames_dir = None
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(case_dir, filename)
                file.save(file_path)
                
                if is_video_file(filename):
                    
                    print(f"Processing video file: {filename}")
                    video_name = os.path.splitext(filename)[0]
                    frames_dir = os.path.join(FRAMES_FOLDER, video_name)
                    
                    
                    frame_paths = extract_frames(file_path, FRAMES_FOLDER)
                    if frame_paths:
                        image_paths.extend(frame_paths)
                        print(f"✅ Extracted {len(frame_paths)} frames from {filename}")
                    else:
                        print(f"⚠️ No frames extracted from {filename}")
                        
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    
                    image_paths.append(file_path)
                    print(f"✅ Added image: {filename}")
        
        return image_paths, frames_dir
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return [], frames_dir

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify system status"""
    status = {
        "status": "running",
        "models_available": MODELS_AVAILABLE,
        "mongodb_connected": collection is not None,
        "missing_files": []
    }
    
    # Check for missing model files
    model_files = [
        "clip_finetuned_few_shot.pth",
        "vit_model.pth", 
        "clip_model_weights.pth",
        "vit_pretrained_evidence.pth"  # Add pretrained ViT model
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            status["missing_files"].append(model_file)
    
    if not MODELS_AVAILABLE:
        status["message"] = "Models not available. Run 'python download_models.py' to set up."
    elif status["missing_files"]:
        status["message"] = f"Missing model files: {', '.join(status['missing_files'])}"
    else:
        status["message"] = "All systems operational"
    
    return jsonify(status)

@app.route("/upload", methods=["POST"])
def upload():
    print("📥 Upload endpoint called")
    
    # Check if models are available
    if not MODELS_AVAILABLE:
        print("❌ Models not available")
        return jsonify({
            "error": "Models not available",
            "message": "Run 'python download_models.py' to set up model files"
        }), 503
    
    print("✅ Models are available")

    if "files" not in request.files:
        print("❌ No 'files' key in request")
        return jsonify({"error": "Use 'files' key for images/videos"}), 400

    files = request.files.getlist("files")
    print(f"📁 Received {len(files)} files")
    
    if len(files) == 0:
        print("❌ No files received")
        return jsonify({"error": "No files received"}), 400

    file_paths = []

    try:
        for file in files:
            if file and file.filename:
                print(f"💾 Saving file: {file.filename}")
                path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                file.save(path)
                file_paths.append(path)
                print(f"✅ Saved: {path}")

        print(f"🎯 Starting prediction for {len(file_paths)} files")
        
        # Process files based on count and type
        if len(file_paths) == 1:
            file_path = file_paths[0]
            print(f"🔍 Single file analysis: {file_path}")
            if is_video_file(file_path):
                print("📹 Video file detected, using predict_multiple_images")
                result = predict_multiple_images(file_path)
            else:
                print("🖼️ Image file detected, using predict_single_image")
                result = predict_single_image(file_path)
        else:
            print(f"📊 Multiple files analysis: {len(file_paths)} files")
            result = predict_multiple_images(file_paths)

        print(f"✅ Prediction completed: {result}")

        # Store result in MongoDB if available
        if collection is not None:
            print("💾 Storing result in MongoDB")
            collection.insert_one({
                "timestamp": datetime.now().isoformat(),
                "result": result
            })
            print("✅ Stored in MongoDB")
        
        # Clean up uploaded files
        for path in file_paths:
            try:
                os.remove(path)
                print(f"🗑️ Cleaned up: {path}")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

        print("🎉 Upload processing completed successfully")
        return jsonify({"message": "Prediction complete", "result": result})
        
    except Exception as e:
        print(f"❌ ERROR in upload endpoint: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        
        # Clean up files on error
        for path in file_paths:
            try:
                os.remove(path)
                print(f"🗑️ Emergency cleanup: {path}")
            except:
                pass
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload/add-to-case', methods=['POST'])
def add_to_case():
    # Check if models are available
    if not MODELS_AVAILABLE:
        return jsonify({
            "error": "Models not available",
            "message": "Run 'python download_models.py' to set up model files"
        }), 503
        
    try:
        email = request.form.get('email')
        case_name = request.form.get('caseName')
        location = request.form.get('location')
        date = request.form.get('date')
        crime_time = request.form.get('crimeTime')
        
        if not email or not case_name:
            return jsonify({'error': 'Email and case name are required'}), 400

        # Create case directory
        case_dir = os.path.join(app.config['UPLOAD_FOLDER'], email, case_name)
        os.makedirs(case_dir, exist_ok=True)

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400

        # Check file types
        video_files = [f for f in files if f.filename and is_video_file(f.filename)]
        
        # Validate upload constraints
        if video_files:
            if len(files) > 1:
                return jsonify({'error': 'Please upload only one video file at a time'}), 400
        else:
            # For images
            if len(files) < 1:
                return jsonify({'error': 'Please upload at least one image'}), 400

        # Process uploaded files
        image_paths, frames_dir = process_uploaded_files(files, case_dir)

        if not image_paths:
            return jsonify({'error': 'No valid image or video files provided'}), 400

        print(f"📊 Processing {len(image_paths)} images/frames for case analysis...")

        # Analyze the crime scene
        analysis_result = process_crime_scene(image_paths)
        
        # Add scene details to result
        if analysis_result.get('status') == 'success':
            analysis_result['analysis']['scene_details'] = {
                'location': location,
                'date': date,
                'time': crime_time
            }
        else:
            analysis_result['scene_details'] = {
                'location': location,
                'date': date,
                'time': crime_time
            }

        # Save result to file
        result_file = os.path.join(case_dir, 'analysis_result.json')
        with open(result_file, 'w') as f:
            json.dump(analysis_result, f)

        # Clean up temporary frames directory but keep case files
        if frames_dir and os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
            except Exception as e:
                print(f"Warning: Could not clean up frames directory: {e}")

        return jsonify({
            'message': 'Files added and analyzed successfully',
            'result': analysis_result
        }), 200

    except Exception as e:
        print(f"Error in add_to_case: {str(e)}") 
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/reanalyze', methods=['POST'])
def reanalyze_case():
    # Check if models are available
    if not MODELS_AVAILABLE:
        return jsonify({
            "error": "Models not available", 
            "message": "Run 'python download_models.py' to set up model files"
        }), 503
        
    try:
        data = request.json
        email = data.get('email')
        case_name = data.get('caseName')
        
        if not email or not case_name:
            return jsonify({'error': 'Email and case name are required'}), 400

        case_dir = os.path.join(app.config['UPLOAD_FOLDER'], email, case_name)
        if not os.path.exists(case_dir):
            return jsonify({'error': 'Case directory not found'}), 400

        # Find all media files in case directory
        all_files = []
        for root, _, files in os.walk(case_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.mov', '.avi')):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            return jsonify({'error': 'No images or videos found in case'}), 400

        # Process files to get image paths
        image_paths = []
        frames_dir = None
        
        for file_path in all_files:
            if is_video_file(file_path):
                # Extract frames from video
                video_name = os.path.splitext(os.path.basename(file_path))[0]
                frames_dir = os.path.join(FRAMES_FOLDER, video_name)
                
                # Extract frames for analysis
                frame_paths = extract_frames(file_path, FRAMES_FOLDER)
                if frame_paths:
                    image_paths.extend(frame_paths)
                    print(f"✅ Re-extracted {len(frame_paths)} frames from {os.path.basename(file_path)}")
            else:
                # Direct image file
                image_paths.append(file_path)

        if not image_paths:
            return jsonify({'error': 'No valid images or video frames found'}), 400

        print(f"📊 Re-analyzing {len(image_paths)} images/frames...")

        # Re-analyze the crime scene
        analysis_result = process_crime_scene(image_paths)

        # Save updated result
        result_file = os.path.join(case_dir, 'analysis_result.json')
        with open(result_file, 'w') as f:
            json.dump(analysis_result, f)

        # Clean up temporary frames directory
        if frames_dir and os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
            except Exception as e:
                print(f"Warning: Could not clean up frames directory: {e}")

        return jsonify({'message': 'Case reanalyzed successfully', 'result': analysis_result}), 200

    except Exception as e:
        print(f"Error in reanalyze_case: {str(e)}") 
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
