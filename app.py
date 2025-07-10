# app.py (เวอร์ชันแยกหน้าจอวิเคราะห์)

import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(layout="wide", page_title="โปรแกรมวิเคราะห์สภาพผิว", page_icon="👩‍⚕️")

# --- ฟังก์ชันโหลดโมเดล (เหมือนเดิม) ---
@st.cache_resource
def load_wound_model():
    processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

@st.cache_resource
def load_acne_model():
    model_name = "imfarzanansari/skintelligent-acne"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

# --- ฟังก์ชันทำนายผล (เหมือนเดิม) ---
def predict(image, processor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    num_classes = model.config.num_labels
    k_value = min(num_classes, 6)
    top_prob, top_catid = torch.topk(probabilities, k_value)
    
    results = []
    for i in range(top_prob.size(1)):
        label = model.config.id2label[top_catid[0][i].item()]
        prob = top_prob[0][i].item()
        results.append({"label": label, "score": prob})
    return results

# --- พจนานุกรมสำหรับแปลภาษา (เหมือนเดิม) ---
wound_translation_dict = {
    "Abrasions": "แผลถลอก", "Bruises": "แผลฟกช้ำ", "Burns": "แผลไฟไหม้/น้ำร้อนลวก",
    "Cut": "แผลบาด / แผลตัด", "Diabetic Wounds": "แผลเบาหวาน", "Laceration": "แผลฉีกขาด",
    "Normal": "ผิวปกติ", "Pressure Wounds": "แผลกดทับ", "Venous Wounds": "แผลจากหลอดเลือดดำ",
    "extravasation-wound-images": "แผลจากการรั่วของยา", "foot-ulcers": "แผลที่เท้า",
    "haemangioma": "ปานแดง", "leg-ulcer-images": "แผลที่ขา", "malignant-wound-images": "แผลมะเร็ง",
    "meningitis": "แผลเยื่อหุ้มสมองอักเสบ", "miscellaneous": "แผลประเภทอื่นๆ",
    "orthopaedic-wounds": "แผลศัลยกรรมกระดูก", "pilonidal-sinus": "ฝีคัณฑสูตร",
    "pressure-ulcer-images-a": "แผลกดทับ (ประเภท A)", "pressure-ulcer-images-b": "แผลกดทับ (ประเภท B)",
    "toes": "แผลที่นิ้วเท้า"
}
acne_translation_dict = {
    "level -1": "ผิวปกติ (Clear Skin)", "level 0": "มีสิวประปราย (Occasional Spots)",
    "level 1": "สิวระดับเล็กน้อย (Mild Acne)", "level 2": "สิวระดับปานกลาง (Moderate Acne)",
    "level 3": "สิวระดับรุนแรง (Severe Acne)", "level 4": "สิวระดับรุนแรงมาก (Very Severe Acne)"
}


# --- ส่วน UI หลัก ---
st.title("โปรแกรมวิเคราะห์สภาพผิวเบื้องต้น 👩‍⚕️")

# ******** 1. สร้างแท็บหลักเพื่อแยกประเภทการวิเคราะห์ ********
tab_wound, tab_acne = st.tabs(["🩹  วิเคราะห์บาดแผล", "😊  วิเคราะห์สิว"])


# --- หน้าจอสำหรับ "วิเคราะห์บาดแผล" ---
with tab_wound:
    st.header("วิเคราะห์ประเภทบาดแผล")
    st.write("อัปโหลดรูปภาพ หรือใช้กล้องถ่ายภาพบาดแผลของคุณ")
    
    # โหลดโมเดลสำหรับบาดแผล
    wound_processor, wound_model = load_wound_model()
    
    # ใช้วิดเจ็ตที่มี key ไม่ซ้ำกัน
    wound_uploader = st.file_uploader("เลือกไฟล์รูปภาพแผล", type=["jpg", "jpeg", "png"], key="wound_uploader")
    wound_camera = st.camera_input("ถ่ายภาพแผล", key="wound_camera")
    wound_image_file = wound_uploader or wound_camera

    if wound_image_file:
        col1, col2 = st.columns(2, gap="large")
        wound_image = Image.open(wound_image_file)

        with col1:
            st.image(wound_image, caption="รูปภาพบาดแผล", use_container_width=True)
            analyze_wound_button = st.button("วิเคราะห์บาดแผล", use_container_width=True, type="primary", key="wound_button")

        with col2:
            if analyze_wound_button:
                with st.spinner("กำลังวิเคราะห์..."):
                    predictions = predict(wound_image, wound_processor, wound_model)
                    with st.container(border=True):
                        top_prediction = predictions[0]
                        thai_label = wound_translation_dict.get(top_prediction['label'], top_prediction['label'])
                        st.metric(label="ประเภทที่เป็นไปได้มากที่สุด", value=thai_label, delta=f"ความมั่นใจ {top_prediction['score']:.2%}")
                    with st.expander("แสดงรายละเอียด 5 อันดับแรก"):
                        for p in predictions:
                            st.write(f"- **{wound_translation_dict.get(p['label'], p['label'])}**: {p['score']:.2%}")


# --- หน้าจอสำหรับ "วิเคราะห์สิว" ---
with tab_acne:
    st.header("วิเคราะห์ระดับความรุนแรงของสิว")
    st.write("อัปโหลดรูปภาพ หรือใช้กล้องถ่ายภาพใบหน้าบริเวณที่มีสิว")

    # โหลดโมเดลสำหรับสิว
    acne_processor, acne_model = load_acne_model()
    
    # ใช้วิดเจ็ตที่มี key ไม่ซ้ำกัน
    acne_uploader = st.file_uploader("เลือกไฟล์รูปภาพสิว", type=["jpg", "jpeg", "png"], key="acne_uploader")
    acne_camera = st.camera_input("ถ่ายภาพสิว", key="acne_camera")
    acne_image_file = acne_uploader or acne_camera

    if acne_image_file:
        col1, col2 = st.columns(2, gap="large")
        acne_image = Image.open(acne_image_file)

        with col1:
            st.image(acne_image, caption="รูปภาพสิว", use_container_width=True)
            analyze_acne_button = st.button("วิเคราะห์สิว", use_container_width=True, type="primary", key="acne_button")

        with col2:
            if analyze_acne_button:
                with st.spinner("กำลังวิเคราะห์..."):
                    predictions = predict(acne_image, acne_processor, acne_model)
                    with st.container(border=True):
                        top_prediction = predictions[0]
                        thai_label = acne_translation_dict.get(top_prediction['label'], top_prediction['label'])
                        st.metric(label="ประเภทที่เป็นไปได้มากที่สุด", value=thai_label, delta=f"ความมั่นใจ {top_prediction['score']:.2%}")
                    with st.expander("แสดงรายละเอียดทั้งหมด"):
                        for p in predictions:
                            st.write(f"- **{acne_translation_dict.get(p['label'], p['label'])}**: {p['score']:.2%}")