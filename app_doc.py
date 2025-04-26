# app.py

import os
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. โหลด environment variables จากไฟล์ .env
load_dotenv(".env")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# 2. เริ่มต้น Qdrant (แบบ In-Memory)
qdrant_client = QdrantClient(":memory:")

# สร้าง Collection สำหรับเก็บเวกเตอร์
qdrant_client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 3. ข้อมูลเอกสารที่กำหนดเอง
documents = [
    # (ข้อมูลเหมือนเดิมที่คุณให้มา)
    "อุทยานแห่งชาติดอยภูคา ที่เที่ยวน่าน อุทยานแห่งชาติดอยภูคา จ.น่าน...",
    "1715 อุทยานแห่งชาติดอยภูคา ที่เที่ยวน่าน จุดชมวิว 1715...",
    "ดอยภูแว : ดอยภูแว ที่เที่ยวน่านธรรมชาติเป็นยอดเขาสูง...",
    "อาหารเช้า&เฉาก๊วยนมสด :วัดพระธาตุเขาน้อย ที่เที่ยวน่านในเมือง...",
    "บ่อเกลือสินเธาว์ : ที่เที่ยวน่าน บ่อเกลือสินเธาว์ นั้นเรียกได้ว่า...",
    "โรงเรียนชาวนาตำบลศิลาเพชร (Farmer School): ที่เที่ยวน่าน ฟาร์มสเตย์...",
    "วัดปรางค์ : ที่เที่ยวน่าน วัดปรางค์ พลาดไม่ได้ หากมาอำเภอปัว...",
    "วัดภูมินทร์ :ที่เที่ยวน่าน วัดภูมินทร์ เป็นวัดอารามหลวงใจกลางเมืองน่าน...",
]


# 4. ฟังก์ชันแปลงข้อความเป็นเวกเตอร์ และเพิ่มลง Qdrant
def add_documents_to_qdrant(documents):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedding_model.encode(documents).tolist()
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": documents[i]})
        for i in range(len(documents))
    ]
    qdrant_client.upsert(collection_name="documents", points=points)


# 5. ฟังก์ชันการค้นหาเอกสาร
def search_documents(query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])[0].tolist()
    search_results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=5,
    )
    return [hit.payload["text"] for hit in search_results]


# 6. ฟังก์ชันสร้างคำตอบด้วย Groq
def generate_answer(query):
    system_prompt = """
    คุณคือ AI chatbot ผู้ช่วยแนะนำสถานที่ท่องเที่ยว
    ให้ตอบเฉพาะข้อมูลที่อยู่ในเอกสารเท่านั้น
    หากผู้ใช้สอบถามสถานที่ที่ไม่อยู่ในจังหวัดน่าน ให้ตอบว่า
    'ขออภัย สถานนั้นไม่มีข้อมูลอยู่'
    """
    temperature = 0.7

    retrieved_docs = search_documents(query)
    context = "\n".join(retrieved_docs)
    full_prompt = (
        f"{system_prompt}\n\nข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {query}\n\nคำตอบ:"
    )

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
    )
    return response.choices[0].message.content


# 7. สร้างอินเทอร์เฟซด้วย Streamlit
def main():
    st.title("RAG Chatbot สำหรับข้อมูล ที่เที่ยวในจังหวัดน่าน")
    st.write("สวัสดี! ฉันคือ Chatbot ที่ช่วยตอบคำถามเกี่ยวกับจังหวัดน่าน")

    # เพิ่มข้อมูลเอกสารลงใน Qdrant
    if st.button("โหลดข้อมูลเอกสาร"):
        add_documents_to_qdrant(documents)
        st.success("ข้อมูลเอกสารพร้อมใช้งานแล้ว!")

    query = st.text_input("คุณ: ", placeholder="พิมพ์คำถามของคุณที่นี่...")

    if st.button("ส่ง"):
        if query:
            answer = generate_answer(query)
            st.write("Bot:", answer)
        else:
            st.warning("กรุณาพิมพ์คำถามก่อนส่ง")


# 8. เรียกใช้แอปพลิเคชัน
if __name__ == "__main__":
    main()
