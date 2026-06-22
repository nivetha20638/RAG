import streamlit as st
import os
import io
import requests
import fitz
from PIL import Image
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from faster_whisper import WhisperModel
import google.generativeai as genai

# ---------------- API CONFIG ----------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------- FLASK EMBEDDING WRAPPER ----------------
class FlaskEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        response = requests.post(
            "http://localhost:5000/embed",
            json={"text": text}
        )
        return response.json()["embedding"]

# ---------------- PDF TEXT ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# ---------------- IMAGE EXTRACTION ----------------
def extract_images_from_pdfs(pdf_docs):
    images = []
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            for img in page.get_images(full=True):
                base_image = doc.extract_image(img[0])
                image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                images.append(image)
    return images

# ---------------- IMAGE CAPTION ----------------
def get_image_summaries(images):
    model = genai.GenerativeModel("gemini-2.0-flash")
    summaries = []
    for img in images:
        response = model.generate_content([
            img,
            "Describe the contents of this image including any text or tables."
        ])
        summaries.append(response.text)
    return summaries

# ---------------- AUDIO ----------------
def get_audio_text(audio_files):
    model = WhisperModel("tiny", device="cpu")
    text = ""
    os.makedirs("audio_uploads", exist_ok=True)

    for audio in audio_files:
        path = f"audio_uploads/{audio.name}"
        with open(path, "wb") as f:
            f.write(audio.read())
        segments, _ = model.transcribe(path)
        text += " ".join(seg.text for seg in segments) + "\n"
        os.remove(path)

    return text

# ---------------- TEXT SPLIT ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ---------------- VECTOR STORE ----------------
def build_vector_store(chunks):
    embeddings = FlaskEmbeddings()
    docs = [Document(page_content=chunk) for chunk in chunks]
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local("faiss_index")

# ---------------- QA CHAIN ----------------
def get_chain():
    prompt_template = """
    Answer clearly using the context. If not found, say "Not available in context."

    Context:
    {context}

    Question:
    {question}
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model

# ---------------- USER QUERY ----------------
def answer_question(question):
    embeddings = FlaskEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    chain = get_chain()
    context = "\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "question": question})
    st.write("Reply:", response.content)

# ---------------- UI ----------------
def main():
    st.set_page_config(page_title="Multimodal RAG")
    st.header("📚 Multimodal RAG: PDF + Image + Audio")

    question = st.text_input("Ask a question")

    if question and os.path.exists("faiss_index"):
        answer_question(question)

    with st.sidebar:
        st.title("Upload Files")
        pdfs = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
        images = st.file_uploader("Images", type=["png", "jpg"], accept_multiple_files=True)
        audios = st.file_uploader("Audio", type="mp3", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Indexing..."):
                text = get_pdf_text(pdfs) if pdfs else ""
                pdf_images = extract_images_from_pdfs(pdfs) if pdfs else []
                uploaded_images = [Image.open(i) for i in images] if images else []
                img_summaries = get_image_summaries(pdf_images + uploaded_images)
                audio_text = get_audio_text(audios) if audios else ""

                all_text = text + "\n".join(img_summaries) + audio_text
                chunks = get_text_chunks(all_text)
                build_vector_store(chunks)

                st.success("Documents indexed!")

if __name__ == "__main__":
    main()
