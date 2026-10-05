import os
import glob
import re
import time
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. ตั้งค่าหน้าเว็บ
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

# 3. เตรียม Vector Store (แคชไว้ด้วย st.cache_resource)
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

    # โมเดลขนาดกะทัดรัด ประหยัด RAM ไม่เกินโควตา Streamlit Cloud
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [item["text"] for item in chunks_data]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
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

# 5. ฟังก์ชันสร้างคำตอบแบบ Auto-Fallback
def generate_rag_answer(query: str, retrieved_chunks: list[dict]):
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

    gemini_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    groq_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")

    # ทางเลือกหลัก: Gemini API
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            candidate_models = ["gemini-3.8-flash", "gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for m in candidate_models:
                try:
                    response = client.models.generate_content(model=m, contents=prompt)
                    if response and response.text:
                        return response.text
                except Exception:
                    continue
        except Exception:
            pass

    # ทางเลือกสำรอง: Groq API
    if groq_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
            candidate_groq_models = ["llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"]
            for gm in candidate_groq_models:
                try:
                    res = groq_client.chat.completions.create(
                        model=gm,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    if res.choices and res.choices[0].message.content:
                        return res.choices[0].message.content
                except Exception:
                    continue
        except Exception:
            pass

    return "⚠️ กรุณาตรวจสอบ GEMINI_API_KEY หรือ GROQ_API_KEY ใน Streamlit Secrets ให้ถูกต้อง"

# ฟังก์ชัน Generator สำหรับจำลอง Streaming ให้ UI ตอบทีละคำอย่างลื่นไหล
def text_streamer(text: str):
    tokens = re.split(r'(\s+)', text)
    for token in tokens:
        yield token
        time.sleep(0.015)

# 6. ส่วนแถบด้านข้าง (Sidebar) แนะนำการใช้งานและตัวอย่างคำถาม
with st.sidebar:
    st.header("🧭 แนะนำการใช้งาน")
    st.write("ระบบผู้ช่วยอัจฉริยะ ตอบคำถามข้อมูลท่องเที่ยวและวัฒนธรรมในจังหวัดปราจีนบุรี โดยดึงข้อมูลจากเอกสารทางการและข้อเท็จจริงในพื้นที่")
    
    st.markdown("---")
    st.subheader("💡 คำถามตัวอย่าง (คลิกเพื่อถาม)")
    
    sample_queries = [
        "น้ำตกเหวนรกเปิดให้เข้าชมกี่โมง และมีค่าธรรมเนียมเข้าชมเท่าไหร่",
        "พระอุโบสถวัดแก้วพิจิตรมีความพิเศษทางสถาปัตยกรรมอย่างไร",
        "เทศกาลล่องแก่งหินเพิงจัดขึ้นช่วงเดือนไหนของปี",
        "ถ้าต้องการซื้อผลิตภัณฑ์สมุนไพรอภัยภูเบศร ซื้อได้ที่ไหน",
        "เกาะเสม็ดมีเรือข้ามฟากกี่โมง"  # คำถามทดสอบ Negative Test (ไม่มีในเอกสาร)
    ]
    
    for sq in sample_queries:
        if st.button(sq, use_container_width=True):
            st.session_state["preset_query"] = sq
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        st.session_state.messages = []
        if "preset_query" in st.session_state:
            del st.session_state["preset_query"]
        st.rerun()

# 7. ส่วนหน้าจอหลัก (Chatbot Interface)
st.title("🧭 ผู้ช่วยท่องเที่ยวและวัฒนธรรมปราจีนบุรี (RAG AI)")
st.caption("🔍 ขับเคลื่อนด้วยเทคนิค RAG (Retrieval-Augmented Generation) ป้องกันข้อมูลมโน (Zero Hallucination)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ฟังก์ชันย่อยสำหรับวาดกล่องเอกสารอ้างอิงให้สวยงาม
def display_sources(sources):
    if sources:
        with st.expander("📚 เอกสารและข้อมูลอ้างอิง"):
            for idx, s in enumerate(sources, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**ลำดับที่ {idx}: `{s['source']}`**")
                with col2:
                    st.markdown(f"`Similarity: {s['score']:.3f}`")
                st.info(f"\"{s['text'][:240]}...\"")

# แสดงประวัติการแชต
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            display_sources(msg["sources"])

# ตรวจสอบว่ามีคำถามที่กดมาจากปุ่ม Sidebar หรือไม่
selected_query = None
if "preset_query" in st.session_state and st.session_state["preset_query"]:
    selected_query = st.session_state.pop("preset_query")

# ช่องรับคำถาม
input_query = st.chat_input("พิมพ์คำถามท่องเที่ยวปราจีนบุรี เช่น น้ำตกเหวนรกเปิดกี่โมง...")
active_query = selected_query or input_query

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("กำลังสืบค้นคลังข้อมูลและเรียบเรียงคำตอบ..."):
            retrieved = retrieve_context(active_query, top_k=3)
            answer = generate_rag_answer(active_query, retrieved)
            
            # แสดงผลแบบ Streaming
            st.write_stream(text_streamer(answer))
            display_sources(retrieved)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })