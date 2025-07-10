# app.py (เวอร์ชันแก้ไข)

import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- คำเตือน ---
st.set_page_config(page_title="Wound Classifier", page_icon="🩹")
st.warning("คำเตือน: นี่เป็นเพียงโปรเจกต์สาธิตและไม่ใช่เครื่องมือทางการแพทย์ ห้ามใช้เพื่อการวินิจฉัยโรค")

@st.cache_resource
def load_model():
    """
    โหลด Processor และ Model จาก Hugging Face
    """
    processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

def predict(image, processor, model):
    """
    รับภาพมาทำนายผลลัพธ์ และคืนค่าเป็น List ของ dict ที่มี label และ score
    """
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

# --- ส่วนหน้าตาของเว็บ (UI) ---

st.title("โปรแกรมวิเคราะห์ประเภทบาดแผลเบื้องต้น 🩹")
st.write("อัปโหลดรูปภาพบาดแผลเพื่อจำแนกประเภทตามโมเดล AI")

# ******** เพิ่มส่วนนี้เข้ามา ********
# สร้าง Dictionary สำหรับการแปลภาษา
translation_dict = {
    "Abrasions": "แผลถลอก",
    "Cut": "แผลบาด / แผลตัด",
    "Normal": "ผิวปกติ",
    "Laceration": "แผลฉีกขาด",
    "Venous Wounds": "แผลจากหลอดเลือดดำ",
    "burns": "แผลไฟไหม้/น้ำร้อนลวก",
    "foot-ulcers": "แผลที่เท้า",
    "leg-ulcer-images": "แผลที่ขา"
    # สามารถเพิ่มคำแปลสำหรับประเภทแผลอื่นๆ ของโมเดลได้ที่นี่
}

try:
    processor, model = load_model()
    uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
        st.write("")

        if st.button("ทำการตรวจสอบ"):
            with st.spinner("กำลังวิเคราะห์... กรุณารอสักครู่"):
                predictions = predict(image, processor, model)
                
                st.success("การวิเคราะห์เสร็จสิ้น!")
                
                # ******** แก้ไขหัวข้อตรงนี้ ********
                st.write("### ผลลัพธ์การวิเคราะห์:")
                
                top_prediction = predictions[0]
                
                # ******** เพิ่มการแปลภาษา ********
                thai_label = translation_dict.get(top_prediction['label'], top_prediction['label'])
                
                st.metric(label=f"ประเภทที่เป็นไปได้มากที่สุด", value=thai_label, delta=f"ความมั่นใจ {top_prediction['score']:.2%}")
                
                st.write("---")
                
                # ******** แก้ไขหัวข้อตรงนี้ ********
                st.write("#### 5 อันดับแรกที่เป็นไปได้:")
                for p in predictions:
                    # ******** เพิ่มการแปลภาษา ********
                    thai_label_list = translation_dict.get(p['label'], p['label'])
                    st.write(f"- **{thai_label_list}**: {p['score']:.2%}")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    st.info("โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตและลองรีเฟรชหน้าเว็บอีกครั้ง")