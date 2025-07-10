import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# 1. โหลด Processor และ Model จาก Hugging Face
processor = AutoImageProcessor.from_pretrained("Hemg/Wound-Image-classification")
model = AutoModelForImageClassification.from_pretrained("Hemg/Wound-Image-classification")

# 2. เตรียมรูปภาพ
# --- ตัวอย่าง: ดึงรูปภาพแผลไฟไหม้จาก URL ---
url = "https://hdmall.co.th/blog/wp-content/uploads/2024/04/home-remedies-painful-blisters.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# --- หรือ: โหลดรูปจากไฟล์ในคอมพิวเตอร์ของคุณ ---
# image_path = "path/to/your/wound_image.jpg"
# image = Image.open(image_path).convert("RGB")

# 3. เตรียมข้อมูลรูปภาพสำหรับโมเดล
# processor จะปรับขนาดและแปลงรูปภาพให้อยู่ในรูปแบบที่โมเดลต้องการ
inputs = processor(images=image, return_tensors="pt")

# 4. ส่งรูปภาพเข้าโมเดลเพื่อทำการจำแนก
# torch.no_grad() ช่วยให้โปรแกรมทำงานเร็วขึ้นเพราะไม่ต้องคำนวณ gradient
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. แปลงผลลัพธ์เป็นความน่าจะเป็นและหาคำตอบที่ดีที่สุด
# softmax จะแปลงคะแนนดิบ (logits) ให้เป็นความน่าจะเป็น (0 ถึง 1)
probabilities = torch.nn.functional.softmax(logits, dim=-1)
# argmax จะหาตำแหน่งของค่าที่สูงที่สุด
predicted_class_idx = probabilities.argmax(-1).item()

# 6. แสดงผลลัพธ์
# model.config.id2label[predicted_class_idx] คือการดึงชื่อประเภทของแผลจากตำแหน่งที่โมเดลทายว่าถูกต้องที่สุด
predicted_class = model.config.id2label[predicted_class_idx]
confidence_score = probabilities.max().item()

print(f"Predicted Class: {predicted_class}")
print(f"Confidence Score: {confidence_score:.4f}")

# แสดง 5 อันดับแรกที่เป็นไปได้
print("\nTop 5 Predictions:")
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(1)):
    label = model.config.id2label[top5_catid[0][i].item()]
    prob = top5_prob[0][i].item()
    print(f"{i+1}. {label}: {prob:.4f}")