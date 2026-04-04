import PyPDF2
import sys

def extract_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    print(f"--- Page {i+1} ---")
                    print(text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_text(sys.argv[1])
