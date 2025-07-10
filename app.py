# app.py (เวอร์ชันแก้ไข ImportError)

import streamlit as st
from PIL import Image
import torch
# ******** 1. ลบ ViltForImageAndTextClassification ออกจาก import ส่วนกลาง ********
from transformers import AutoProcessor

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(layout="wide", page_title="โปรแกรมวิเคราะห์สภาพผิว", page_icon="👩‍⚕️")

# --- ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_wound_model():
    from transformers import AutoModelForImageClassification
    processor = AutoProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

@st.cache_resource
def load_acne_model():
    from transformers import AutoModelForImageClassification
    model_name = "imfarzanansari/skintelligent-acne"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

@st.cache_resource
def load_dermatology_model():
    # ******** 2. ย้าย ViltForImageAndTextClassification มา import ที่นี่ ********
    from transformers import ViltForImageAndTextClassification
    model_name = "dandelin/vilt-b32-finetuned-dermnet"
    processor = AutoProcessor.from_pretrained(model_name)
    model = ViltForImageAndTextClassification.from_pretrained(model_name)
    return processor, model

# --- ฟังก์ชันทำนายผล ---
def predict(image, processor, model, text_input="a photo of a skin condition"):
    # โมเดล Vilt ต้องการการตรวจสอบประเภทที่แม่นยำขึ้น
    # เราจึงต้อง import คลาสของมันมาเพื่อใช้ isinstance
    from transformers import ViltForImageAndTextClassification

    if image.mode != "RGB":
        image = image.convert("RGB")

    if isinstance(model, ViltForImageAndTextClassification):
        inputs = processor(images=image, text=text_input, return_tensors="pt")
    else: 
        inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    num_classes = model.config.num_labels
    k_value = min(num_classes, 7)
    top_prob, top_catid = torch.topk(probabilities, k_value)
    
    results = []
    for i in range(top_prob.size(1)):
        label = model.config.id2label[top_catid[0][i].item()]
        prob = top_prob[0][i].item()
        results.append({"label": label, "score": prob})
    return results

# --- พจนานุกรมสำหรับแปลภาษา (เหมือนเดิม) ---
wound_translation_dict = {
    "Abrasions": "แผลถลอก", "Bruises": "แผลฟกช้ำ", "Burns": "แผลไฟไหม้/น้ำร้อนลวก", "Cut": "แผลบาด / แผลตัด", "Diabetic Wounds": "แผลเบาหวาน", "Laceration": "แผลฉีกขาด", "Normal": "ผิวปกติ", "Pressure Wounds": "แผลกดทับ", "Venous Wounds": "แผลจากหลอดเลือดดำ", "extravasation-wound-images": "แผลจากการรั่วของยา", "foot-ulcers": "แผลที่เท้า", "haemangioma": "ปานแดง", "leg-ulcer-images": "แผลที่ขา", "malignant-wound-images": "แผลมะเร็ง", "meningitis": "แผลเยื่อหุ้มสมองอักเสบ", "miscellaneous": "แผลประเภทอื่นๆ", "orthopaedic-wounds": "แผลศัลยกรรมกระดูก", "pilonidal-sinus": "ฝีคัณฑสูตร", "pressure-ulcer-images-a": "แผลกดทับ (ประเภท A)", "pressure-ulcer-images-b": "แผลกดทับ (ประเภท B)", "toes": "แผลที่นิ้วเท้า"
}
acne_translation_dict = {
    "level -1": "ผิวปกติ (Clear Skin)", "level 0": "มีสิวประปราย (Occasional Spots)", "level 1": "สิวระดับเล็กน้อย (Mild Acne)", "level 2": "สิวระดับปานกลาง (Moderate Acne)", "level 3": "สิวระดับรุนแรง (Severe Acne)", "level 4": "สิวระดับรุนแรงมาก (Very Severe Acne)"
}
dermatology_translation_dict = {
    'actinic keratosis basal cell carcinoma and other malignant lesions': 'โรคผิวหนังก่อนเป็นมะเร็งและมะเร็งผิวหนัง', 'acne and rosacea photos': 'สิวและโรคหน้าแดง', 'atopic dermatitis photos': 'โรคผื่นภูมิแพ้ผิวหนัง', 'bullous disease photos': 'โรคผิวหนังตุ่มน้ำพอง', 'cellulitis impetigo and other bacterial infections': 'การติดเชื้อแบคทีเรียที่ผิวหนัง', 'eczema photos': 'โรคผื่นแพ้ผิวหนัง (Eczema)', 'exanthems and drug eruptions': 'ผื่นจากเชื้อไวรัสและผื่นแพ้ยา', 'hair loss photos alopecia and other hair diseases': 'ผมร่วงและโรคเกี่ยวกับเส้นผม', 'herpes hpv and other stds photos': 'โรคเริม, HPV และโรคติดต่อทางเพศสัมพันธ์', 'light diseases and disorders of pigmentation': 'โรคจากแสงและภาวะสีผิวผิดปกติ', 'lupus and other connective tissue diseases': 'โรคแพ้ภูมิตัวเอง (Lupus)', 'melanoma skin cancer nevi and moles': 'มะเร็งผิวหนังเมลาโนมา, ไฝ', 'nail fungus and other nail disease': 'เชื้อราที่เล็บและโรคเล็บ', 'poison ivy photos and other contact dermatitis': 'ผื่นแพ้สัมผัส', 'psoriasis pictures lichen planus and related diseases': 'โรคสะเก็ดเงินและโรคผิวหนังอักเสบเรื้อรัง', 'scabies lyme disease and other infestations and bites': 'โรคหิด, โรคไลม์, แมลงกัดต่อย', 'seborrheic keratoses and other benign tumors': 'เนื้องอกที่ไม่ใช่มะเร็ง (เช่น ติ่งเนื้อ)', 'systemic disease': 'โรคทางระบบที่แสดงอาการทางผิวหนัง', 'tinea ringworm candidiasis and other fungal infections': 'การติดเชื้อรา (กลาก, เกลื้อน)', 'urticaria hives': 'ลมพิษ', 'vascular tumors': 'เนื้องอกของหลอดเลือด', 'vasculitis photos': 'โรคหลอดเลือดอักเสบ', 'warts molluscum and other viral infections': 'หูด, หูดข้าวสุก และการติดเชื้อไวรัสอื่น'
}

# --- UI หลักของแอป (ส่วนนี้ไม่มีการแก้ไข) ---
with st.sidebar:
    st.title("ℹ️ เกี่ยวกับแอปพลิเคชัน")
    st.info("แอปพลิเคชันนี้ใช้โมเดล AI เพื่อจำแนกประเภทของสภาพผิวเบื้องต้นจากรูปภาพ")
    st.warning("**คำเตือน:** นี่ไม่ใช่เครื่องมือทางการแพทย์ และห้ามใช้เพื่อการวินิจฉัยโรคโดยเด็ดขาด")
st.title("โปรแกรมวิเคราะห์สภาพผิวเบื้องต้น 👩‍⚕️")
analysis_type = st.selectbox("เลือกประเภทการวิเคราะห์:", ("วิเคราะห์บาดแผล", "วิเคราะห์สิว", "วิเคราะห์โรคผิวหนังทั่วไป"))
st.write(f"คุณเลือก: **{analysis_type}** | กรุณาอัปโหลดรูปภาพ หรือใช้กล้องถ่ายภาพ")
try:
    if analysis_type == "วิเคราะห์บาดแผล":
        processor, model = load_wound_model()
        translation_dict = wound_translation_dict
    elif analysis_type == "วิเคราะห์สิว":
        processor, model = load_acne_model()
        translation_dict = acne_translation_dict
    else: 
        processor, model = load_dermatology_model()
        translation_dict = dermatology_translation_dict
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    st.info("โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ตและลองรีเฟรชหน้าเว็บนี้อีกครั้ง")
    st.stop()
tab1, tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายภาพ"])
with tab1:
    uploaded_file = st.file_uploader("ลากและวางไฟล์ที่นี่ หรือกดปุ่มเพื่อเลือกไฟล์", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with tab2:
    camera_file = st.camera_input("กดปุ่มเพื่อเริ่มใช้งานกล้อง")
image_file = uploaded_file or camera_file
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
                    thai_label = translation_dict.get(top_prediction['label'].lower(), top_prediction['label'])
                    st.metric(label="ประเภทที่เป็นไปได้มากที่สุด", value=thai_label, delta=f"ความมั่นใจ {top_prediction['score']:.2%}")
                with st.expander(f"แสดงรายละเอียดอันดับที่เป็นไปได้"):
                    for p in predictions:
                        thai_label_list = translation_dict.get(p['label'].lower(), p['label'])
                        st.write(f"- **{thai_label_list}**: {p['score']:.2%}")
        else:
            st.info("กรุณากดปุ่ม 'ทำการวิเคราะห์' เพื่อดูผลลัพธ์")