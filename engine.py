import cv2
import re
import os
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

# Try to import multiple OCR engines
OCR_ENGINES = []

# Try PaddleOCR
try:
    from paddleocr import PaddleOCR
    OCR_ENGINES.append('paddle')
    print("✓ PaddleOCR available")
except:
    print("✗ PaddleOCR not available")

# Try EasyOCR (often better for English text)
try:
    import easyocr
    OCR_ENGINES.append('easy')
    print("✓ EasyOCR available")
except:
    print("✗ EasyOCR not available (install: pip install easyocr)")

# Try Tesseract
try:
    import pytesseract
    OCR_ENGINES.append('tesseract')
    print("✓ Tesseract available")
except:
    print("✗ Tesseract not available (install: pip install pytesseract)")

# --- INITIALIZATION ---
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-nano-aadhar-card", filename="model.pt")
detection_model = YOLO(model_path)

# Initialize OCR engines
ocr_engines = {}
if 'paddle' in OCR_ENGINES:
    ocr_engines['paddle'] = PaddleOCR(lang='en')
if 'easy' in OCR_ENGINES:
    ocr_engines['easy'] = easyocr.Reader(['en'], gpu=False)
if 'tesseract' in OCR_ENGINES:
    ocr_engines['tesseract'] = None  # pytesseract doesn't need initialization

def preprocess_for_ocr(img, preset='default'):
    """Multiple preprocessing presets for different OCR engines"""
    
    if preset == 'default':
        # Light preprocessing - good for high quality images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    elif preset == 'high_contrast':
        # Aggressive contrast enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    elif preset == 'adaptive':
        # Adaptive thresholding - good for varying lighting
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    elif preset == 'morph':
        # Morphological operations - good for noisy images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    
    else:
        return img

def run_paddle_ocr(img):
    """Run PaddleOCR and extract text"""
    try:
        result = ocr_engines['paddle'].predict(img)
        texts = []
        if result and len(result) > 0:
            for item in result[0].get('rec_text', []):
                if item['score'] > 0.5:
                    texts.append(item['text'])
        return texts
    except Exception as e:
        print(f"  PaddleOCR error: {e}")
        return []

def run_easy_ocr(img):
    """Run EasyOCR and extract text"""
    try:
        result = ocr_engines['easy'].readtext(img)
        texts = [text for (bbox, text, conf) in result if conf > 0.5]
        return texts
    except Exception as e:
        print(f"  EasyOCR error: {e}")
        return []

def run_tesseract_ocr(img):
    """Run Tesseract and extract text"""
    try:
        # Convert to grayscale for Tesseract
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return [line.strip() for line in text.split('\n') if line.strip()]
    except Exception as e:
        print(f"  Tesseract error: {e}")
        return []

def extract_text_multi_engine(img, region_name="UNKNOWN"):
    """Use EasyOCR and Tesseract for best quality Aadhar extraction"""
    all_results = []
    
    # Use EasyOCR (best quality for Aadhar cards)
    if 'easy' in ocr_engines:
        texts = run_easy_ocr(img)
        if texts:
            all_results.extend(texts)
        
        # Try high contrast version
        processed = preprocess_for_ocr(img, 'high_contrast')
        texts = run_easy_ocr(processed)
        if texts:
            all_results.extend(texts)
    
    # Also use Tesseract for additional coverage
    if 'tesseract' in ocr_engines:
        texts = run_tesseract_ocr(img)
        if texts:
            all_results.extend(texts)
    
    return all_results

def extract_details(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ Error: Could not read image")
        return None
    
    # --- DETECTION ---
    results = detection_model(image_path, verbose=False)
    detected_regions = {}
    
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_name = results[0].names[int(box.cls[0])]
            
            h, w, _ = img.shape
            y1_pad = max(0, y1 - 15)
            x1_pad = max(0, x1 - 15)
            y2_pad = min(h, y2 + 15)
            x2_pad = min(w, x2 + 15)
            
            cropped = img[y1_pad:y2_pad, x1_pad:x2_pad]
            detected_regions[class_name] = cropped
    else:
        detected_regions['FULL_IMAGE'] = img
    
    # --- OCR EXTRACTION ---
    all_text = []
    
    for region_name, region_img in detected_regions.items():
        texts = extract_text_multi_engine(region_img, region_name)
        all_text.extend(texts)
    
    full_text = " ".join(all_text).upper()
    
    # --- PARSING ---
    # Aadhaar Number
    num_clean = re.sub(r'[^\d]', '', full_text)
    aadhaar = None
    
    # Try to find 12 consecutive digits
    match = re.search(r'\d{12}', num_clean)
    if match:
        aadhaar = match.group(0)
    else:
        # Try formatted versions
        match = re.search(r'\d{4}[\s-]*\d{4}[\s-]*\d{4}', full_text)
        if match:
            aadhaar = re.sub(r'\D', '', match.group(0))
    
    # DOB
    dob = None
    dob_patterns = [
        r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b',
        r'\b(19\d{2}|20\d{2})\b'
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, full_text)
        if match:
            dob = match.group(0)
            break
    
    # Gender - Fixed logic
    gender = "NOT FOUND"
    if "FEMALE" in full_text:
        gender = "FEMALE"
    elif "MALE" in full_text:
        gender = "MALE"
    
    # Name - Improved extraction for Indian names
    name = "Not Found"
    
    # Remove common Aadhar card text to find the actual name
    ignore_words = {
        'GOVERNMENT', 'OF', 'INDIA', 'UNIQUE', 'IDENTIFICATION', 'AUTHORITY', 
        'DOB', 'DATE', 'BIRTH', 'GENDER', 'MALE', 'FEMALE', 'YOB', 'YEAR', 
        'ADDRESS', 'PIN', 'CODE', 'PHOTO', 'SIGNATURE', 'ISSUE', 'DATE', 
        'NAME', 'DIST', 'REPUBLIC', 'CARDS', 'SERVICES'
    }
    
    # Get all words with at least 2 chars (more flexible pattern)
    all_words = re.findall(r'\b[A-Z][A-Za-z]{2,}\b', full_text)
    
    # Filter out ignored words
    filtered_words = [w for w in all_words if w.upper() not in ignore_words and len(w) > 2]
    
    # First, try to find "Name:" or similar patterns
    name_patterns = [
        r'NAME[:\s]+([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)',
        r'Name\s+of\s+([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, full_text)
        if match:
            potential = match.group(1).strip()
            if len(potential) >= 5:
                name = potential.title()
                break
    
    # If still not found, look for the first 2-3 capitalized words that form a name
    if name == "Not Found" and len(filtered_words) >= 2:
        # Skip likely header words at the beginning
        start_idx = 0
        for i, word in enumerate(filtered_words[:8]):
            # Skip common patterns at the start (GOVERNMENT OF INDIA, etc.)
            if word.upper() in ['GOVERNMENT', 'OF', 'INDIA', 'UNIQUE', 'IDENTIFICATION', 'SERVICES']:
                start_idx = i + 1
            else:
                break
        
        # Try 2-word name first
        for i in range(start_idx, min(start_idx + 3, len(filtered_words) - 1)):
            word1 = filtered_words[i]
            word2 = filtered_words[i + 1] if i + 1 < len(filtered_words) else ""
            
            # Skip if either word looks like a header or number
            if len(word1) < 3 or len(word2) < 3:
                continue
            
            # Combine and check if it looks like a name (not all caps, has variety)
            potential = f"{word1} {word2}"
            
            # Skip if it looks like a header phrase
            if potential.upper() in ['INDIA GOVT', 'GOVT OF', 'UNIQUE IDENTIFICATION', 'IDENTIFICATION AUTHORITY']:
                continue
            
            if len(potential) >= 5:
                name = potential.title()
                break
        
        # If 2-word name didn't work, try 3 words
        if name == "Not Found" and len(filtered_words) >= start_idx + 3:
            for i in range(start_idx, min(start_idx + 2, len(filtered_words) - 2)):
                word1 = filtered_words[i]
                word2 = filtered_words[i + 1]
                word3 = filtered_words[i + 2]
                
                if len(word1) < 3 or len(word2) < 3 or len(word3) < 3:
                    continue
                
                potential = f"{word1} {word2} {word3}"
                if len(potential) >= 7:
                    name = potential.title()
                    break
    
    # If still not found, use first 2 valid words
    if name == "Not Found" and len(filtered_words) >= 2:
        name = ' '.join(filtered_words[:2]).title()
    
    # Clean up name if it contains artifacts
    if name != "Not Found":
        name = re.sub(r'\b(Name|Number|No|Do|ID)\b', '', name).strip()
        name = re.sub(r'\s+', ' ', name).strip()
        # Remove trailing/leading special characters
        name = re.sub(r'^[^\w]+|[^\w]+$', '', name)
        if not name:
            name = "Not Found"
    
    return {
        "Aadhaar Number": aadhaar if aadhaar else "Not Found",
        "Name": name,
        "Date of Birth": dob if dob else "Not Found",
        "Gender": gender
    }

# --- EXECUTION ---
if __name__ == "__main__":
    if not OCR_ENGINES:
        print("\n❌ ERROR: No OCR engines available!")
        print("Install at least one: pip install easyocr  OR  pip install pytesseract")
        exit(1)
    
    MY_IMAGE = "/Users/sujit/Desktop/runtime_terrorists/1cfb4eca9083a70e4d0a60963bb729d9.jpg"
    
    if os.path.exists(MY_IMAGE):
        print("\n" + "=" * 60)
        print("🔍 EXTRACTING AADHAAR CARD DETAILS...")
        print("=" * 60)
        
        data = extract_details(MY_IMAGE)
        
        if data:
            print("\n📊 EXTRACTION RESULTS:\n")
            print(f"  Aadhaar Number : {data['Aadhaar Number']}")
            print(f"  Name           : {data['Name']}")
            print(f"  Date of Birth  : {data['Date of Birth']}")
            print(f"  Gender         : {data['Gender']}")
        else:
            print("\n❌ Failed to process image.")
        
        print("\n" + "=" * 60 + "\n")
    else:
        print(f"❌ Error: Could not find image file")