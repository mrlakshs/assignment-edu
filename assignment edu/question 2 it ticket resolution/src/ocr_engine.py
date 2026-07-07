import pytesseract
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from screenshots to normalize the data stream.
    Requires Tesseract-OCR installed on the host machine.
    """
    try:
        logging.info(f"Running OCR on {image_path}")
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        
        
        clean_text = " ".join(extracted_text.split())
        return clean_text
    
    except Exception as e:
        logging.error(f"OCR failed for {image_path}: {e}")
        return ""