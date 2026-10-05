import os
import glob
import re
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Prachinburi Travel Guide (RAG)",
    page_icon="🧭",
    layout="wide"
)

# 2. ฟังก์ชันทำความสะอาดและแบ่ง Chunk
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_document(text: str, chunk_size: int = 450, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 3. เตรียมคลังความรู้และ Vector Store (โหลดครั้งเดียวด้วย cache_resource)
@st.cache_resource(show_spinner="กำลังโหลดฐานข้อมูลท่องเที่ยวปราจีนบุรี...")
def init_vector_store():
    data_files = glob.glob("data/*.txt")
    if not data_files:
        return None, None, None

    chunks_data = []
    for filepath in data_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        cleaned = clean_text(content)
        file_chunks = chunk_document(cleaned)
        for idx, ch in enumerate(file_chunks):
            chunks_data.append({
                "source": filename,
                "chunk_id": f"{filename}_chunk_{idx+1}",
                "text": ch
            })

    # โมเดลขนาดเล็ก รองรับภาษาไทย ประหยัด RAM ไม่เกินโควตา Streamlit Cloud
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [item["text"] for item in chunks_data]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product บน Normalized Vectors เทียบเท่า Cosine Similarity
    index.add(np.array(embeddings, dtype=np.float32))

    return embed_model, index, chunks_data

embed_model, index, chunks_data = init_vector_store()

# 4. ฟังก์ชันค้นหาบริบท (Retrieval)
def retrieve_context(query: str, top_k: int = 3):
    if not index or not embed_model:
        return []
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_vec, dtype=np.float32), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks_data):
            results.append({
                "source": chunks_data[idx]["source"],
                "text": chunks_data[idx]["text"],
                "score": float(score)
            })
    return results

# 5. ฟังก์ชันสร้างคำตอบด้วย Groq LLM API
def generate_rag_answer(query: str, retrieved_chunks: list[dict], api_key: str):
    client = Groq(api_key=api_key)
    
    context_str = "\n\n".join(
        [f"[แหล่งที่มา: {c['source']}]\n{c['text']}" for c in retrieved_chunks]
    )

    prompt = f"""คุณเป็นไกด์และผู้ช่วยแนะนำข้อมูลการท่องเที่ยวและวัฒนธรรมประจำจังหวัดปราจีนบุรี

กฎการตอบคำถาม:
1. ตอบคำถามโดยใช้เฉพาะข้อมูลจาก "บริบทที่กำหนดให้" ด้านล่างนี้เท่านั้น ห้ามแต่งคำตอบขึ้นมาเอง
2. หากข้อมูลในบริบทไม่มีคำตอบ หรือข้อมูลไม่เพียงพอ ให้ตอบตรงๆ ว่า "ไม่พบข้อมูลนี้ในเอกสารคลังความรู้"
3. อธิบายคำตอบอย่างสุภาพ กระชับ ถูกต้อง และระบุแหล่งที่มา (ชื่อไฟล์เอกสาร) ท้ายคำตอบเสมอ

บริบท:
{context_str}

คำถาม: {query}
คำตอบ:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# 6. ส่วนติดต่อผู้ใช้ (Chatbot UI)
st.title("🧭 ผู้ช่วยท่องเที่ยวและวัฒนธรรมปราจีนบุรี (RAG AI)")
st.caption("สอบถามแหล่งท่องเที่ยว วัด ประวัติศาสตร์ เทศกาล ร้านอาหาร และการเดินทางในจังหวัดปราจีนบุรี")

# อ่าน API Key จาก Secrets ของ Streamlit Cloud
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY ใน Streamlit Secrets กรุณาตั้งค่าก่อนใช้งาน")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการแชต
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 เอกสารอ้างอิง"):
                for s in msg["sources"]:
                    st.write(f"- **{s['source']}** (Similarity Score: {s['score']:.3f})")
                    st.caption(f"\"{s['text'][:160]}...\"")

# ช่องรับคำถาม
if user_query := st.chat_input("พิมพ์คำถามท่องเที่ยวปราจีนบุรี เช่น น้ำตกเหวนรกเปิดกี่โมง..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ค้นหาและสร้างคำตอบ
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลในคลังเอกสารและสรุปคำตอบ..."):
            retrieved = retrieve_context(user_query, top_k=3)
            answer = generate_rag_answer(user_query, retrieved, api_key)
            st.markdown(answer)
            
            if retrieved:
                with st.expander("📚 เอกสารอ้างอิง"):
                    for s in retrieved:
                        st.write(f"- **{s['source']}** (Similarity Score: {s['score']:.3f})")
                        st.caption(f"\"{s['text'][:160]}...\"")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })