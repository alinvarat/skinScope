# app.py (เวอร์ชันปรับปรุงภาษาไทย)

import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(layout="wide", page_title="โปรแกรมวิเคราะห์บาดแผล", page_icon="🩹")

# --- ฟังก์ชันโหลดโมเดลและทำนายผล (เหมือนเดิม) ---
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

def predict(image, processor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    results = []
    for i in range(top5_prob.size(1)):
        label = model.config.id2label[top5_catid[0][i].item()]
        prob = top5_prob[0][i].item()
        results.append({"label": label, "score": prob})
    return results

# --- พจนานุกรมสำหรับแปลภาษา ---
translation_dict = {
    "Abrasions": "แผลถลอก",
    "Cut": "แผลบาด / แผลตัด",
    "Normal": "ผิวปกติ",
    "Laceration": "แผลฉีกขาด",
    "Venous Wounds": "แผลจากหลอดเลือดดำ",
    "burns": "แผลไฟไหม้/น้ำร้อนลวก",
    "foot-ulcers": "แผลที่เท้า",
    "leg-ulcer-images": "แผลที่ขา",
    # เพิ่มคำแปลสำหรับประเภทแผลอื่นๆ ของโมเดลได้ที่นี่
    "extravasation-wound-images": "แผลจากการรั่วของยา",
    "haemangioma": "ปานแดง",
    "malignant-wound-images": "แผลมะเร็ง",
    "meningitis": "แผลเยื่อหุ้มสมองอักเสบ",
    "miscellaneous": "แผลประเภทอื่นๆ",
    "orthopaedic-wounds": "แผลศัลยกรรมกระดูก",
    "pilonidal-sinus": "ฝีคัณฑสูตร",
    "pressure-ulcer-images-a": "แผลกดทับ (ประเภท A)",
    "pressure-ulcer-images-b": "แผลกดทับ (ประเภท B)",
    "toes": "แผลที่นิ้วเท้า"
}

# --- แถบด้านข้าง (Sidebar) ---
with st.sidebar:
    st.title("ℹ️ เกี่ยวกับแอปพลิเคชัน")
    st.info(
        "แอปพลิเคชันนี้ใช้โมเดลปัญญาประดิษฐ์ (AI) เพื่อจำแนกประเภทของบาดแผลเบื้องต้นจากรูปภาพ "
        "สร้างขึ้นเพื่อเป็นโปรเจกต์สาธิตการใช้งาน Streamlit และ Hugging Face"
    )
    st.warning("**คำเตือน:** นี่ไม่ใช่เครื่องมือทางการแพทย์ และห้ามใช้เพื่อการวินิจฉัยโรคโดยเด็ดขาด")

# --- หน้าหลักของแอป ---
st.title("โปรแกรมวิเคราะห์ประเภทบาดแผลเบื้องต้น 🩹")
st.write("เลือกอัปโหลดรูปภาพ หรือใช้กล้องถ่ายภาพบาดแผลของคุณ เพื่อให้ AI ลองวิเคราะห์เบื้องต้น")

# โหลดโมเดล
try:
    processor, model = load_model()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    st.info("โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตและลองรีเฟรชหน้าเว็บอีกครั้ง")
    st.stop()

# สร้างแท็บสำหรับเลือกวิธีการส่งรูปภาพ
tab1, tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายภาพ"])

with tab1:
    uploaded_file = st.file_uploader(
        "ลากและวางไฟล์ที่นี่ หรือกดปุ่มเพื่อเลือกไฟล์", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

with tab2:
    camera_file = st.camera_input("กดปุ่มเพื่อเริ่มใช้งานกล้อง")

image_file = uploaded_file or camera_file

# แสดงผลลัพธ์เมื่อมีรูปภาพ
if image_file is not None:
    col1, col2 = st.columns(2, gap="large")
    image = Image.open(image_file)

    with col1:
        st.subheader("🖼️ รูปภาพของคุณ")
        st.image(image, use_container_width=True)
        analyze_button = st.button("ทำการวิเคราะห์", use_container_width=True, type="primary")

    with col2:
        st.subheader("📈 ผลการวิเคราะห์")
        if analyze_button:
            with st.spinner("กำลังวิเคราะห์... กรุณารอสักครู่"):
                predictions = predict(image, processor, model)
                
                with st.container(border=True):
                    top_prediction = predictions[0]
                    thai_label = translation_dict.get(top_prediction['label'], top_prediction['label'])
                    
                    st.metric(
                        label="ประเภทที่เป็นไปได้มากที่สุด",
                        value=thai_label,
                        delta=f"ความมั่นใจ {top_prediction['score']:.2%}"
                    )
                
                with st.expander("แสดงรายละเอียด 5 อันดับแรก"):
                    for p in predictions:
                        thai_label_list = translation_dict.get(p['label'], p['label'])
                        st.write(f"- **{thai_label_list}**: {p['score']:.2%}")
        else:
            st.info("กรุณากดปุ่ม 'ทำการวิเคราะห์' เพื่อดูผลลัพธ์")