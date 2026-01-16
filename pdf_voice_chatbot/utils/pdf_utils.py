from pypdf import PdfReader
from io import BytesIO

def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        reader = PdfReader(BytesIO(file.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
