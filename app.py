from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure your custom logic is imported correctly
try:
    from engine import extract_details
except ImportError:
    print("[!] Error: engine.py not found. Make sure it's in the same folder.")

app = Flask(__name__)

# 1. ENHANCED CORS: Allows your frontend (Vite/React/Live URL) to talk to this backend
CORS(app)

# Configure temporary upload storage
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def health_check():
    """Simple route to check if the server is awake via browser."""
    return jsonify({
        "status": "online",
        "message": "ML Backend is running successfully",
        "environment": "Production" if os.environ.get("PORT") else "Development"
    }), 200

@app.route('/extract', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "is_valid_ocr": False}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file", "is_valid_ocr": False}), 400

    # Save the file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # TRIGGER ENGINE
        data = extract_details(filepath)
        
        # Validation Logic
        has_name = data.get('Name') != "Not Found"
        has_aadhar = data.get('Aadhaar Number') != "Not Found"
        data['is_valid_ocr'] = bool(has_name and has_aadhar)
        
        return jsonify(data)

    except Exception as e:
        print(f"[!] System Error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    finally:
        # CLEANUP: Ensure sensitive images are deleted immediately
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    # DYNAMIC PORT: Render assigns a port via environment variable
    # Local runs will fall back to 5000
    port = int(os.environ.get("PORT", 5000))
    
    # host='0.0.0.0' is mandatory for cloud visibility
    # debug=False is safer and uses less RAM in production
    app.run(host='0.0.0.0', port=port, debug=False)
