import pdfplumber

def extract_text_from_pdf(pdf_path):
    """pulls all text from a pdf file, page by page"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

    # sometimes pdfs have weird whitespace issues so we clean it up a bit
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    # quick test - change the path to any pdf you have
    import sys
    if len(sys.argv) > 1:
        result = extract_text_from_pdf(sys.argv[1])
        print(result[:500])  # just print first 500 chars to check
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
