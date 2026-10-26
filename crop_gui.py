"""
Single-file Crop Disease Detection (Tiny built-in dataset + Tkinter GUI).

What this does:
- Creates a tiny synthetic dataset (4-20 small images) of 'Healthy' and 'Diseased' leaves.
  * Healthy = mostly green images
  * Diseased = green with brown spots
- Trains a small CNN from scratch on that dataset (fast - fits on CPU).
- Provides a Tkinter GUI:
  * Train Model
  * Load Model
  * Predict (open image) -> displays image + predicted class + probability

Notes:
- This is a toy/demo implementation so it runs without external datasets or downloads.
- For real accuracy use a real dataset (e.g., PlantVillage) and a bigger model.
"""

import os
import random
import io
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------- Config ----------
DATA_DIR = "dataset_tiny"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_PATH = "tiny_cnn.pth"
CLASS_FILE = "classes_small.json"
IMG_SIZE = 96           # small images to speed up training
EPOCHS = 12
BATCH_SIZE = 8
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
# ----------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

classes = ["Healthy", "Diseased"]

# ---------- Synthetic dataset generation ----------
def make_healthy_image(size=IMG_SIZE):
    # green background with slight variations
    base = np.zeros((size, size, 3), dtype=np.uint8)
    # green-ish background
    g = np.random.randint(110, 200)
    r = np.random.randint(10, 50)
    b = np.random.randint(10, 50)
    base[:, :] = (r, g, b)
    img = Image.fromarray(base)
    # add some light veins-ish lines
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(3):
        x0 = np.random.randint(0, w//2)
        y0 = np.random.randint(0, h)
        x1 = np.random.randint(w//2, w)
        y1 = np.random.randint(0, h)
        draw.line((x0, y0, x1, y1), fill=(r+10, g+10, b+10), width=1)
    return img

def make_diseased_image(size=IMG_SIZE):
    img = make_healthy_image(size)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # add brown spots
    num_spots = np.random.randint(3, 8)
    for _ in range(num_spots):
        rx = np.random.randint(6, 20)
        ry = np.random.randint(6, 20)
        x = np.random.randint(0, w-rx)
        y = np.random.randint(0, h-ry)
        # brown color variance
        br = np.random.randint(90, 160)
        bg = np.random.randint(40, 100)
        bb = np.random.randint(10, 40)
        draw.ellipse((x, y, x+rx, y+ry), fill=(br, bg, bb))
    # add some irregular brown patches/lines
    for _ in range(2):
        x0 = np.random.randint(0, w)
        y0 = np.random.randint(0, h)
        x1 = x0 + np.random.randint(-10, 10)
        y1 = y0 + np.random.randint(-10, 10)
        draw.line((x0, y0, x1, y1), fill=(120, 70, 20), width=2)
    return img

def create_dataset(train_count_per_class=10, val_count_per_class=4):
    if os.path.exists(DATA_DIR):
        return  # don't recreate if exists
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    # create train images
    for i in range(train_count_per_class):
        img = make_healthy_image() if i % 2 == 0 else make_healthy_image()
        img.save(os.path.join(TRAIN_DIR, "Healthy", f"healthy_{i}.png"))
    for i in range(train_count_per_class):
        img = make_diseased_image()
        img.save(os.path.join(TRAIN_DIR, "Diseased", f"diseased_{i}.png"))

    # create val images
    for i in range(val_count_per_class):
        img = make_healthy_image()
        img.save(os.path.join(VAL_DIR, "Healthy", f"healthy_val_{i}.png"))
    for i in range(val_count_per_class):
        img = make_diseased_image()
        img.save(os.path.join(VAL_DIR, "Diseased", f"diseased_val_{i}.png"))

    print(f"Created synthetic dataset at '{DATA_DIR}'")

# ---------- Simple dataset loader ----------
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SimpleFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = pil_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- Small CNN model ----------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 48 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 24 -> 12
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1) # 12 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------- Training and utilities ----------
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def train_model(progress_callback=None):
    print("Preparing data...")
    train_ds = SimpleFolderDataset(TRAIN_DIR, transform=transform_train)
    val_ds = SimpleFolderDataset(VAL_DIR, transform=transform_eval)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset not found or empty. Make sure dataset exists.")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyCNN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        # validation
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)
        val_acc = val_corrects / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} train_acc: {epoch_acc:.3f} val_acc: {val_acc:.3f}")
        if progress_callback:
            progress_callback(f"Epoch {epoch+1}/{EPOCHS} - val_acc: {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training finished. Best val acc: {best_acc:.3f}")
    return model

def load_model_if_exists():
    model = TinyCNN(num_classes=len(classes))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        return None

# ---------- GUI ----------
class App:
    def __init__(self, master):
        self.master = master
        master.title("Crop Disease Detection (Tiny Demo)")
        master.geometry("640x480")
        self.model = None

        # Top frame for buttons/status
        frame = tk.Frame(master)
        frame.pack(padx=10, pady=6, anchor='nw')

        self.train_btn = tk.Button(frame, text="Create Dataset & Train", command=self.on_train, width=22)
        self.train_btn.grid(row=0, column=0, padx=4)
        self.load_btn = tk.Button(frame, text="Load Trained Model", command=self.on_load, width=18)
        self.load_btn.grid(row=0, column=1, padx=4)
        self.predict_btn = tk.Button(frame, text="Predict Image", command=self.on_predict, width=18)
        self.predict_btn.grid(row=0, column=2, padx=4)

        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        self.status_label = tk.Label(master, textvariable=self.status_text, anchor='w')
        self.status_label.pack(fill='x', padx=10)

        # Canvas for image
        self.canvas = tk.Canvas(master, width=320, height=320, bg='gray90')
        self.canvas.pack(padx=10, pady=10)

        # Prediction label
        self.pred_var = tk.StringVar()
        self.pred_var.set("Prediction: -")
        self.pred_label = tk.Label(master, textvariable=self.pred_var, font=("Arial", 14))
        self.pred_label.pack(pady=4)

        # If a trained model exists, load it
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model_if_exists()
                self.status_text.set("Loaded existing model.")
            except Exception as e:
                print("Could not load existing model:", e)

    def set_status(self, txt):
        self.status_text.set(txt)
        self.master.update_idletasks()

    def on_train(self):
        try:
            self.set_status("Creating synthetic dataset (if missing)...")
            create_dataset(train_count_per_class=12, val_count_per_class=5)
            self.set_status("Training model (this may take ~30s on CPU)...")
            # run training and update status periodically
            def progress_cb(msg):
                self.set_status(msg)
            model = train_model(progress_callback=progress_cb)
            self.model = model
            self.set_status("Training complete. Model saved.")
            messagebox.showinfo("Training", "Training complete. Model saved to " + MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error during training.")

    def on_load(self):
        try:
            model = load_model_if_exists()
            if model is None:
                messagebox.showwarning("Load Model", "No saved model found. Train first.")
                self.set_status("No model found.")
            else:
                self.model = model
                self.set_status("Model loaded.")
                messagebox.showinfo("Load Model", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error loading model.")

    def preprocess_pil(self, pil_img):
        img = pil_img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        t = transform_eval(img)
        t = t.unsqueeze(0)  # batch
        return t

    def on_predict(self):
        if self.model is None:
            messagebox.showwarning("No model", "No model loaded. Train or load a model first.")
            return
        filepath = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not filepath:
            return
        pil_img = Image.open(filepath).convert("RGB")
        # show image in canvas (resize to fit)
        disp = pil_img.copy()
        disp.thumbnail((320,320))
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(160, 160, image=self.photo)

        # preprocess and infer
        with torch.no_grad():
            inp = self.preprocess_pil(pil_img).to(DEVICE)
            out = self.model(inp)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_conf = float(probs[pred_idx])
            label = classes[pred_idx]
            self.pred_var.set(f"Prediction: {label}  (Conf: {pred_conf:.2f})")
            self.set_status(f"Predicted: {label} (conf {pred_conf:.2f})")

# ---------- Main ----------
if __name__ == "__main__":
    # Create dataset folder if missing
    create_dataset(train_count_per_class=12, val_count_per_class=5)

    root = tk.Tk()
    app = App(root)
    root.mainloop()
