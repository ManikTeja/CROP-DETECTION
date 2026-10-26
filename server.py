from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io

app = Flask(__name__)
CORS(app)  # allow connection from React frontend

# Define the model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 64 * 64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 8 * 64 * 64)
        x = self.fc(x)
        return x

# Load model safely
model = TinyCNN()
try:
    model.load_state_dict(torch.load("tiny_cnn.pt", weights_only=True))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# Define label names for predictions
classes = ["Healthy", "Blight", "Rust", "Mildew"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
