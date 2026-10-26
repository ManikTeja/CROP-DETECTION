import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms
import random

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = 96
MODEL_PATH = "crop_model.pth"
CLASSES = ["Healthy", "Diseased"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# SYNTHETIC DATA CREATION
# -------------------------------
def make_healthy_image(size=IMG_SIZE):
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, :] = (np.random.randint(20, 60), np.random.randint(140, 200), np.random.randint(20, 60))
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        draw.line(
            (random.randint(0, size), random.randint(0, size),
             random.randint(0, size), random.randint(0, size)),
            fill=(0, random.randint(150, 200), 0),
            width=2)
    return img

def make_diseased_image(size=IMG_SIZE):
    img = make_healthy_image(size)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        x, y = random.randint(0, size-15), random.randint(0, size-15)
        r = random.randint(5, 15)
        draw.ellipse((x, y, x+r, y+r), fill=(139, 69, 19))
    return img

def create_dataset(n=30):
    data, labels = [], []
    for _ in range(n):
        data.append(make_healthy_image())
        labels.append(0)
        data.append(make_diseased_image())
        labels.append(1)
    return data, labels

# -------------------------------
# MODEL
# -------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_model():
    st.info("Creating synthetic dataset...")
    data, labels = create_dataset(40)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    X = torch.stack([transform(img) for img in data])
    y = torch.tensor(labels)
    
    model = TinyCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    st.info("Training model... ⏳")
    for epoch in range(8):
        optimizer.zero_grad()
        outputs = model(X.to(DEVICE))
        loss = criterion(outputs, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(1).cpu().numpy()
        acc = (preds == np.array(labels)).mean()
        st.progress((epoch+1)/8)
        st.write(f"Epoch {epoch+1}/8 - Loss: {loss.item():.4f} - Acc: {acc:.2f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    st.success("✅ Training complete! Model saved.")
    return model

# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    model = TinyCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        st.warning("Model not found, please train first.")
        return None

# -------------------------------
# PREDICT
# -------------------------------
def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="🌾 Crop Disease Detection", page_icon="🌿", layout="wide")

st.title("🌿 Crop Disease Detection Using Deep Learning")
st.markdown("Detect whether a crop leaf is **Healthy** or **Diseased** using a CNN model. This demo generates a small built-in dataset and trains a model locally.")

col1, col2 = st.columns(2)

with col1:
    st.header("🔧 Train Model")
    if st.button("🚀 Train on Built-in Dataset"):
        model = train_model()

    if st.button("📂 Load Existing Model"):
        model = load_model()
        if model:
            st.success("Model loaded successfully!")

with col2:
    st.header("🧠 Predict Crop Disease")
    uploaded = st.file_uploader("Upload a leaf image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
        
        model = load_model()
        if model:
            probs = predict_image(model, image)
            pred_idx = int(np.argmax(probs))
            st.subheader(f"Prediction: **{CLASSES[pred_idx]}** 🌱")
            st.write(f"Confidence: {probs[pred_idx]*100:.2f}%")

st.markdown("---")
st.markdown("💡 *Demo project by Deep Learning — Crop Disease Detection (Streamlit GUI)*")
