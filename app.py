# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- คำเตือน ---
st.set_page_config(page_title="Wound Classifier", page_icon="🩹")
st.warning("คำเตือน: นี่เป็นเพียงโปรเจกต์สาธิตและไม่ใช่เครื่องมือทางการแพทย์ ห้ามใช้เพื่อการวินิจฉัยโรค")

# ใช้ @st.cache_resource เพื่อให้โหลดโมเดลแค่ครั้งแรกครั้งเดียว ทำให้เว็บเร็วขึ้น
@st.cache_resource
def load_model():
    """
    โหลด Processor และ Model จาก Hugging Face
    ฟังก์ชันนี้จะทำงานแค่ครั้งเดียวและเก็บผลลัพธ์ไว้ใน Cache
    """
    processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

def predict(image, processor, model):
    """
    รับภาพมาทำนายผลลัพธ์ และคืนค่าเป็น List ของ dict ที่มี label และ score
    """
    # ตรวจสอบว่าภาพเป็น RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # เตรียมข้อมูลรูปภาพสำหรับโมเดล
    inputs = processor(images=image, return_tensors="pt")

    # ส่งรูปภาพเข้าโมเดลเพื่อทำการจำแนก
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # แปลงผลลัพธ์เป็นความน่าจะเป็น
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # จัดอันดับ 5 ประเภทที่เป็นไปได้มากที่สุด
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

# โหลดโมเดล (จะดึงจาก cache ถ้าเคยโหลดแล้ว)
try:
    processor, model = load_model()

    # สร้าง UI สำหรับอัปโหลดไฟล์
    uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # แสดงรูปภาพที่อัปโหลด
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
        st.write("") # เพิ่มบรรทัดว่าง

        # สร้างปุ่มสำหรับเริ่มการตรวจสอบ
        if st.button("ทำการตรวจสอบ"):
            # แสดงสถานะว่ากำลังทำงาน
            with st.spinner("กำลังวิเคราะห์... กรุณารอสักครู่"):
                # เรียกใช้ฟังก์ชันทำนาย
                predictions = predict(image, processor, model)
                
                # แสดงผลลัพธ์
                st.success("การวิเคราะห์เสร็จสิ้น!")
                
                st.write("### ผลลัพธ์การทำนาย:")
                
                # แสดงผลอันดับ 1 ให้เด่นชัด
                top_prediction = predictions[0]
                st.metric(label=f"ประเภทที่เป็นไปได้มากที่สุด", value=top_prediction['label'], delta=f"ความมั่นใจ {top_prediction['score']:.2%}")
                
                # แสดง 5 อันดับแรก
                st.write("---")
                st.write("#### 5 อันดับแรกที่เป็นไปได้:")
                for p in predictions:
                    st.write(f"- **{p['label']}**: {p['score']:.2%}")
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    st.info("โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตและลองรีเฟรชหน้าเว็บอีกครั้ง")