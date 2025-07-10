# app.py (เวอร์ชันปรับปรุง UI)

import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- 1. ตั้งค่าหน้าเว็บให้เป็นแบบ Wide Mode ---
st.set_page_config(layout="wide", page_title="Wound Classifier", page_icon="🩹")

# --- Model Loading & Prediction Functions (เหมือนเดิม) ---
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

# --- Translation Dictionary (เหมือนเดิม) ---
translation_dict = {
    "Abrasions": "แผลถลอก",
    "Cut": "แผลบาด / แผลตัด",
    "Normal": "ผิวปกติ",
    "Laceration": "แผลฉีกขาด",
    "Venous Wounds": "แผลจากหลอดเลือดดำ",
    "burns": "แผลไฟไหม้/น้ำร้อนลวก",
    "foot-ulcers": "แผลที่เท้า",
    "leg-ulcer-images": "แผลที่ขา"
}

# --- 2. ย้ายข้อมูลเกี่ยวกับแอปไปไว้ที่ Sidebar ---
with st.sidebar:
    st.title("ℹ️ เกี่ยวกับแอปพลิเคชัน")
    st.info(
        "แอปพลิเคชันนี้ใช้โมเดล AI เพื่อจำแนกประเภทของบาดแผลเบื้องต้นจากรูปภาพ "
        "สร้างขึ้นเพื่อเป็นโปรเจกต์สาธิตการใช้งาน Streamlit และ Hugging Face"
    )
    st.warning("**คำเตือน:** ไม่ใช่เครื่องมือทางการแพทย์ ห้ามใช้เพื่อวินิจฉัยโรค")

# --- Main Page UI ---
st.title("โปรแกรมวิเคราะห์ประเภทบาดแผลเบื้องต้น 🩹")
st.write("อัปโหลดรูปภาพบาดแผลของคุณที่นี่ เพื่อให้ AI ลองวิเคราะห์เบื้องต้น")

# Load model
try:
    processor, model = load_model()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader(
    "เลือกรูปภาพ...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

# --- 3. แบ่งหน้าจอเป็น 2 คอลัมน์เมื่อมีรูปภาพ ---
if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="large")
    image = Image.open(uploaded_file)

    with col1:
        st.subheader("🖼️ รูปภาพของคุณ")
        st.image(image, use_container_width=True)
        analyze_button = st.button("ทำการวิเคราะห์", use_container_width=True, type="primary")

    with col2:
        st.subheader("📈 ผลการวิเคราะห์")
        if analyze_button:
            with st.spinner("กำลังวิเคราะห์..."):
                predictions = predict(image, processor, model)
                
                # --- 4. ใช้ Container และ Expander จัดระเบียบผลลัพธ์ ---
                with st.container(border=True):
                    top_prediction = predictions[0]
                    thai_label = translation_dict.get(top_prediction['label'], top_prediction['label'])
                    
                    st.metric(
                        label="ประเภทที่เป็นไปได้มากที่สุด",
                        value=thai_label,
                        delta=f"ความมั่นใจ {top_prediction['score']:.2%}"
                    )
                
                with st.expander("ดู 5 อันดับแรกที่เป็นไปได้"):
                    for p in predictions:
                        thai_label_list = translation_dict.get(p['label'], p['label'])
                        st.write(f"- **{thai_label_list}**: {p['score']:.2%}")
        else:
            st.info("กรุณากดปุ่ม 'ทำการวิเคราะห์' เพื่อดูผลลัพธ์")