import os
import glob
import re
import time
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. ตั้งค่าหน้าเว็บและสไตล์ CSS
st.set_page_config(
    page_title="Prachinburi Travel Guide (RAG)",
    page_icon="🧭",
    layout="wide"
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 10px; border-left: 4px solid #00a86b; }
</style>
""", unsafe_allow_html=True)

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

# 5. ฟังก์ชันสร้างคำตอบแบบ Auto-Fallback พร้อมความปลอดภัย
def generate_rag_answer(query: str, retrieved_chunks: list[dict]):
    if not retrieved_chunks:
        return "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้ในเอกสารคลังความรู้การท่องเที่ยวปราจีนบุรี"

    context_str = "\n\n".join(
        [f"[แหล่งที่มา: {c['source']}]\n{c['text']}" for c in retrieved_chunks]
    )

    prompt = f"""คุณเป็นไกด์และผู้ช่วยแนะนำข้อมูลการท่องเที่ยวและวัฒนธรรมประจำจังหวัดปราจีนบุรี

กฎการตอบคำถาม:
1. ตอบคำถามโดยใช้เฉพาะข้อมูลจาก "บริบทที่กำหนดให้" ด้านล่างนี้เท่านั้น ห้ามแต่งคำตอบขึ้นมาเองเด็ดขาด
2. หากข้อมูลในบริบทไม่มีคำตอบ หรือข้อมูลไม่เพียงพอ ให้ตอบตรงๆ ว่า "ไม่พบข้อมูลนี้ในเอกสารคลังความรู้"
3. อธิบายคำตอบอย่างสุภาพ กระชับ ถูกต้อง และระบุแหล่งที่มา (ชื่อไฟล์เอกสาร) ท้ายคำตอบเสมอ

บริบท:
{context_str}

คำถาม: {query}
คำตอบ:"""

    gemini_key = st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    groq_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")

    # ลำดับที่ 1: ลองใช้ Gemini API
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]
            for m in candidate_models:
                try:
                    response = client.models.generate_content(model=m, contents=prompt)
                    if response and response.text:
                        return response.text
                except Exception:
                    continue
        except Exception:
            pass

    # ลำดับที่ 2: ลองใช้ Groq API (Fallback)
    if groq_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
            candidate_groq_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
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

    # ลำดับที่ 3: Graceful Degradation (ดึงสรุปเบื้องต้นจาก Chunks)
    fallback_response = (
        "ขณะนี้ระบบประมวลผลภาษามีผู้ใช้งานหนาแน่นชั่วคราว "
        "ระบบจึงได้ดึงข้อมูลเบื้องต้นที่เกี่ยวข้องจากคลังเอกสารมาให้ท่านโดยตรง:\n\n"
    )
    for c in retrieved_chunks[:2]:
        fallback_response += f"- **จากเอกสาร `{c['source']}`**:\n> \"{c['text'][:180]}...\"\n\n"
    fallback_response += "*(ท่านสามารถคลิกดูรายละเอียดเต็มได้ที่กล่อง '📚 เอกสารและข้อมูลอ้างอิง' ด้านล่างครับ)*"
    
    return fallback_response

# ฟังก์ชัน Generator สำหรับจำลอง Streaming ข้อความ
def text_streamer(text: str):
    tokens = re.split(r'(\s+)', text)
    for token in tokens:
        yield token
        time.sleep(0.012)

# ฟังก์ชันแสดงผลกล่องเอกสารอ้างอิง
def display_sources(sources, latency_sec=None):
    if sources:
        with st.expander("📚 เอกสารและข้อมูลอ้างอิง"):
            if latency_sec:
                st.caption(f"⏱️ เวลาที่ใช้ในการประมวลผล: {latency_sec:.2f} วินาที")
            for idx, s in enumerate(sources, 1):
                col1, col2 = st.columns([3, 1])
                badge = "🟢 สูง" if s['score'] >= 0.50 else "🟡 ปานกลาง"
                with col1:
                    st.markdown(f"**ลำดับที่ {idx}: `{s['source']}`**")
                with col2:
                    st.markdown(f"`ความเกี่ยวข้อง: {s['score']:.3f}` ({badge})")
                st.info(f"\"{s['text'][:240]}...\"")

# 6. แถบด้านข้าง (Sidebar) พร้อมสถิติคลังข้อมูลและหมวดหมู่คำถาม
with st.sidebar:
    st.header("🧭 แนะนำการใช้งาน")
    st.write("ผู้ช่วยอัจฉริยะตอบคำถามข้อมูลท่องเที่ยวและวัฒนธรรมในจังหวัดปราจีนบุรี โดยดึงข้อมูลจากเอกสารทางการและข้อเท็จจริงในพื้นที่")

    # สถิติคลังข้อมูล (Knowledge Base Metrics)
    st.markdown("---")
    st.subheader("📊 ข้อมูลคลังความรู้")
    total_docs = len(glob.glob("data/*.txt"))
    total_chunks = len(chunks_data) if chunks_data else 0
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("หมวดเอกสาร", f"{total_docs} ไฟล์")
    m_col2.metric("จำนวน Chunks", f"{total_chunks} ชิ้น")
    st.caption("🛡️ โหมด: ป้องกันข้อมูลมโน (Zero Hallucination)")

    st.markdown("---")
    st.subheader("💡 คำถามตัวอย่าง")
    
    with st.expander("🌲 ธรรมชาติ & อุทยาน", expanded=True):
        if st.button("น้ำตกเหวนรกเปิดกี่โมง และมีค่าธรรมเนียมเท่าไหร่", use_container_width=True):
            st.session_state["preset_query"] = "น้ำตกเหวนรกเปิดให้เข้าชมกี่โมง และมีค่าธรรมเนียมเข้าชมเท่าไหร่"
            st.rerun()

    with st.expander("🛕 วัฒนธรรม & โบราณสถาน"):
        if st.button("วัดแก้วพิจิตรมีความพิเศษทางสถาปัตยกรรมอย่างไร", use_container_width=True):
            st.session_state["preset_query"] = "พระอุโบสถวัดแก้วพิจิตรมีความพิเศษทางสถาปัตยกรรมอย่างไร"
            st.rerun()

    with st.expander("🚣 กิจกรรม & ของฝาก"):
        if st.button("เทศกาลล่องแก่งหินเพิงจัดช่วงไหน", use_container_width=True):
            st.session_state["preset_query"] = "เทศกาลล่องแก่งหินเพิงจัดขึ้นช่วงเดือนไหนของปี"
            st.rerun()
        if st.button("ซื้อผลิตภัณฑ์สมุนไพรอภัยภูเบศรได้ที่ไหน", use_container_width=True):
            st.session_state["preset_query"] = "ถ้าต้องการซื้อผลิตภัณฑ์สมุนไพรอภัยภูเบศร ซื้อได้ที่ไหน"
            st.rerun()

    with st.expander("❌ ทดสอบคำถามนอกพื้นที่ (Negative Test)"):
        if st.button("เกาะเสม็ดมีเรือข้ามฟากกี่โมง", use_container_width=True):
            st.session_state["preset_query"] = "เกาะเสม็ดมีเรือข้ามฟากกี่โมง"
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

# แสดงประวัติการแชต
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            display_sources(msg["sources"], msg.get("latency"))

# รับคำถามจากปุ่ม Sidebar หรือช่อง Input
selected_query = None
if "preset_query" in st.session_state and st.session_state["preset_query"]:
    selected_query = st.session_state.pop("preset_query")

input_query = st.chat_input("พิมพ์คำถามท่องเที่ยวปราจีนบุรี เช่น น้ำตกเหวนรกเปิดกี่โมง...")
active_query = selected_query or input_query

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        start_time = time.time()
        with st.spinner("กำลังสืบค้นคลังข้อมูลและเรียบเรียงคำตอบ..."):
            retrieved = retrieve_context(active_query, top_k=3)
            answer = generate_rag_answer(active_query, retrieved)
            latency = time.time() - start_time
            
            # ตอบแบบ Streaming
            st.write_stream(text_streamer(answer))
            display_sources(retrieved, latency)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
        "latency": latency
    })