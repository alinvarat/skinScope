# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification, ViTForImageClassification

st.set_page_config(layout="wide", page_title="Skin Scope", page_icon="👩‍⚕️")

@st.cache_resource
def load_wound_model():
    processor = AutoProcessor.from_pretrained("Hemg/Wound-Image-classification")
    model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")
    return processor, model

@st.cache_resource
def load_acne_model():
    model_name = "imfarzanansari/skintelligent-acne"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

@st.cache_resource
def load_dermatology_model():
    model_name = "Jayanth2002/vit_base_patch16_224-finetuned-SkinDisease"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

def predict(image, processor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    num_classes = model.config.num_labels
    k_value = min(num_classes, 7)
    top_prob, top_catid = torch.topk(probabilities, k_value)

    label_id_map = getattr(model.config, "id2label", {i: f"Class {i}" for i in range(num_classes)})
    results = []
    for i in range(top_prob.size(1)):
        label = label_id_map[top_catid[0][i].item()]
        prob = top_prob[0][i].item()
        results.append({"label": label, "score": prob})
    return results

with st.sidebar:
    st.title("ℹ️ About Skin Scope")
    st.info("Skin Scope uses AI models to classify basic skin conditions from images.")
    st.warning("**Disclaimer:** This is not a medical tool and should not be used for diagnosis.")

st.title("Skin Scope 👩‍⚕️")
analysis_type = st.selectbox(
    "Select Analysis Type:",
    ("Wound Analysis", "Acne Analysis", "General Skin Disease Analysis")
)
st.write(f"You selected: **{analysis_type}** | Please upload an image or take a photo.")

try:
    if analysis_type == "Wound Analysis":
        processor, model = load_wound_model()
    elif analysis_type == "Acne Analysis":
        processor, model = load_acne_model()
    else:
        processor, model = load_dermatology_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please check your internet connection and try refreshing this page again.")
    st.stop()

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])
with tab1:
    uploaded_file = st.file_uploader("Drag and drop file here or click to select file", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=analysis_type)
with tab2:
    camera_file = st.camera_input("Click to start camera", key=f"cam_{analysis_type}")

image_file = uploaded_file or camera_file

if image_file is not None:
    col1, col2 = st.columns(2, gap="large")
    image = Image.open(image_file)
    with col1:
        st.subheader("🖼️ Your Image")
        st.image(image, use_container_width=True)
        analyze_button = st.button("Perform Analysis", use_container_width=True, type="primary")
    with col2:
        st.subheader("📈 Analysis Results")
        if analyze_button:
            with st.spinner("Analyzing... Please wait"):
                predictions = predict(image, processor, model)
                with st.container(border=True):
                    top_prediction = predictions[0]
                    st.metric(label="Most Probable Type", value=top_prediction['label'], delta=f"{top_prediction['score']*100:.1f}%")

                with st.expander("Show Possible Details"):
                    for p in predictions:
                        st.write(f"- **{p['label']}**: {p['score']:.2%}")
        else:
            st.info("Please click 'Perform Analysis' to see the results.")