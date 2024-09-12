import os
from flask import Flask, render_template, request, redirect, send_from_directory
import pdfplumber
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from fpdf import FPDF
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/summary_files/'

# Load pre-trained models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Use GPU


def pdf_to_text_plumber(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def divide_by_semantics_with_length(text, threshold=0.6, max_words=1000, min_words=400):
    sentences = text.split('. ')
    embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = sentences[0]
    
    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i-1])
        current_word_count = len(current_chunk.split())

        if similarity < threshold or current_word_count + len(sentences[i].split()) > max_words:
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[i]
            else:
                current_chunk += '. ' + sentences[i]
        else:
            current_chunk += '. ' + sentences[i]
    
    if len(current_chunk.split()) >= min_words:
        chunks.append(current_chunk.strip())
    
    return chunks


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def clean_chunks(chunks):
    return [clean_text(chunk) for chunk in chunks]


def summarize_chunks(chunks):
    summaries = []
    for chunk in chunks:
        chunk_length = len(chunk.split())
        if chunk_length > 50:
            try:
                summary = summarizer(chunk, max_length=1500, min_length=20, do_sample=False, clean_up_tokenization_spaces=True)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append(chunk)
        else:
            summaries.append(chunk)
    return summaries


def overall_summary(summaries):
    structured_summary = ""
    for i, summary in enumerate(summaries, 1):
        structured_summary += summary + "\n\n"
    return structured_summary


def strip_unicode(text):
    return text.encode('latin-1', 'ignore').decode('latin-1')

class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Book Summary', ln=True, align='C')
            self.ln(10)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_text(self, text):
        self.add_page()
        self.chapter_body(text)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the file: Extract text and summarize
            book_text = pdf_to_text_plumber(file_path)
            semantic_chunks = divide_by_semantics_with_length(book_text)
            cleaned_semantic_chunks = clean_chunks(semantic_chunks)
            summarized_chunks = summarize_chunks(cleaned_semantic_chunks)
            final_summary = overall_summary(summarized_chunks)

            # Create PDF summary
            pdf = PDF()
            cleaned_summary = strip_unicode(final_summary)
            pdf.add_text(cleaned_summary)
            summary_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'summary.pdf')
            pdf.output(summary_pdf_path)

            # Return the generated summary PDF
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'summary.pdf', as_attachment=True)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)



