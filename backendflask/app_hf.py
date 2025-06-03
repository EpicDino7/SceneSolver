import gradio as gr
import os
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Try to import the crime pipeline
try:
    from crime_pipeline_few_shot import load_models, predict_single_image, predict_multiple_images, process_crime_scene, extract_frames
    MODELS_AVAILABLE = True
    print("✅ Models imported successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load models - {e}")
    MODELS_AVAILABLE = False

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
        MODELS_AVAILABLE = False

def analyze_crime_scene(files):
    """Analyze uploaded crime scene files."""
    if not MODELS_AVAILABLE:
        return "❌ Models not available. Please check model files."
    
    if not files:
        return "❌ No files uploaded."
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            # Save uploaded files
            for file in files:
                if file is not None:
                    filename = secure_filename(os.path.basename(file.name))
                    file_path = os.path.join(temp_dir, filename)
                    shutil.copy2(file.name, file_path)
                    file_paths.append(file_path)
            
            if not file_paths:
                return "❌ No valid files found."
            
            # Process files
            if len(file_paths) == 1:
                result = predict_single_image(file_paths[0])
            else:
                result = predict_multiple_images(file_paths)
            
            # Format result for display
            if isinstance(result, dict):
                formatted_result = json.dumps(result, indent=2)
            else:
                formatted_result = str(result)
            
            return f"✅ Analysis Complete:\n\n{formatted_result}"
            
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"

def health_check():
    """Check system health."""
    status = {
        "models_available": MODELS_AVAILABLE,
        "system": "operational"
    }
    return json.dumps(status, indent=2)

# Create Gradio interface
with gr.Blocks(title="SceneSolver - Crime Scene Analysis") as app:
    gr.Markdown("# 🔍 SceneSolver - Crime Scene Analysis")
    gr.Markdown("Upload crime scene images or videos for AI-powered analysis.")
    
    with gr.Tab("Crime Scene Analysis"):
        files_input = gr.File(
            file_count="multiple",
            file_types=["image", "video"],
            label="Upload Crime Scene Files (Images/Videos)"
        )
        analyze_btn = gr.Button("🔍 Analyze Crime Scene", variant="primary")
        result_output = gr.Textbox(
            label="Analysis Results",
            lines=20,
            placeholder="Analysis results will appear here..."
        )
        
        analyze_btn.click(
            fn=analyze_crime_scene,
            inputs=[files_input],
            outputs=[result_output]
        )
    
    with gr.Tab("System Health"):
        health_btn = gr.Button("🏥 Check System Health")
        health_output = gr.Textbox(
            label="System Status",
            lines=10,
            placeholder="System health information will appear here..."
        )
        
        health_btn.click(
            fn=health_check,
            outputs=[health_output]
        )

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 