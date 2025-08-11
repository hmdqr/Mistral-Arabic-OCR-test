import os
import sys
import base64
import csv
import time
import logging
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv() 


# Configuration
DOC_DIR = "doc"
DB_CSV = "processed_files.csv"
LOG_FILE = "conversion.log"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # in seconds

# Initialize logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

# Ensure API key is set via environment variable
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    print("Error: MISTRAL_API_KEY environment variable not set.")
    sys.exit(1)

client = Mistral(api_key=API_KEY)

FIELDNAMES = ['filename', 'status', 'attempts', 'error']


def load_processed():
    """Load processed file records from CSV into a dict."""
    processed = {}
    if os.path.exists(DB_CSV):
        with open(DB_CSV, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                processed[row['filename']] = row
    return processed


def append_to_db(record):
    """Append a processing record to the CSV database."""
    file_exists = os.path.exists(DB_CSV)
    with open(DB_CSV, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def get_pdf_files():
    """List all PDF files in the DOC_DIR folder."""
    if not os.path.isdir(DOC_DIR):
        print(f"Error: Directory '{DOC_DIR}' not found.")
        sys.exit(1)
    return [f for f in os.listdir(DOC_DIR) if f.lower().endswith('.pdf')]


def encode_pdf(pdf_path):
    """Encode the PDF file to a base64 string."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to encode {pdf_path}: {e}")
        return None


def convert_pdf_to_markdown(pdf_filename):
    """Perform OCR on the PDF and write the output as a markdown file."""
    full_path = os.path.join(DOC_DIR, pdf_filename)
    b64 = encode_pdf(full_path)
    if not b64:
        raise RuntimeError("PDF encoding failed.")

    # Call Mistral OCR
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}"
        },
        include_image_base64=False
    )

    # Write markdown
    output_name = pdf_filename.rsplit('.', 1)[0] + '.md'
    with open(output_name, 'w', encoding='utf-8') as md_file:
        for page in response.pages:
            md_file.write(f"## Page {page.index + 1}\n\n")
            md_file.write(page.markdown + "\n\n")


def main():
    processed = load_processed()
    all_files = get_pdf_files()
    total = len(all_files)
    succeeded = sum(1 for r in processed.values() if r['status'] == 'success')
    to_do = [f for f in all_files if processed.get(f, {}).get('status') != 'success']

    print(f"Found {total} PDF files in '{DOC_DIR}/'. {succeeded} already converted. {len(to_do)} remaining.")

    converted_count = 0
    for idx, pdf in enumerate(to_do, start=1):
        print(f"[{idx}/{len(to_do)}] Processing: {pdf}")
        attempts = 0
        backoff = INITIAL_BACKOFF
        success = False

        while attempts < MAX_RETRIES and not success:
            attempts += 1
            try:
                convert_pdf_to_markdown(pdf)
                success = True
                converted_count += 1
                append_to_db({'filename': pdf, 'status': 'success', 'attempts': attempts, 'error': ''})
                print(f"Success: {pdf} (attempt {attempts})")
                print(f"Warming for the next File...")
                time.sleep(3)
            except Exception as e:
                error_msg = str(e)
                append_to_db({'filename': pdf, 'status': 'error', 'attempts': attempts, 'error': error_msg})
                logging.error(f"{pdf} attempt {attempts} failed: {error_msg}")
                print(f"Error converting {pdf} on attempt {attempts}: {error_msg}")
                if attempts < MAX_RETRIES:
                    print(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2

        if not success:
            print(f"Failed: {pdf} after {attempts} attempts.")

    print(f"\nConversion complete. Total successful conversions: {converted_count} out of {len(to_do)}.")


if __name__ == '__main__':
    main()
