from pypdf import PdfReader
from io import BytesIO

def extract_text_from_pdfs(files):
    """
    Optimized PDF text extraction with efficient string concatenation.
    """
    texts = []  # Use list for efficient concatenation
    for file in files:
        try:
            # Reset file pointer in case it was read before
            file.seek(0)
            file_bytes = file.read()
            reader = PdfReader(BytesIO(file_bytes))
            
            # Extract all pages at once for better performance
            page_texts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_texts.append(page_text.strip())
            
            if page_texts:
                texts.extend(page_texts)
                
        except Exception as e:
            raise ValueError(f"Error reading PDF file {file.name}: {str(e)}")
    
    # Efficient join
    text = "\n".join(texts)
    
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF files.")
    
    return text