# Shared OCR utilities for all report types 
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def load_document(file_path_or_bytes):
    if isinstance(file_path_or_bytes, str):
        if file_path_or_bytes.lower().endswith(('.png', '.jpg', '.jpeg')):
            return DocumentFile.from_images(file_path_or_bytes)
        else:
            return DocumentFile.from_pdf(file_path_or_bytes)
    else:
        # Assume bytes-like object is an image
        return DocumentFile.from_images(file_path_or_bytes)

def run_ocr(doc):
    model = ocr_predictor(pretrained=True)
    result = model(doc)
    return result.export() 