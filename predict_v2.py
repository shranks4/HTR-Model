import torch
import cv2
import numpy as np
from htr_model_v2 import HTRModelV2, NUM_CLASSES, idx2char, greedy_decode, MAX_DECODE_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    model = HTRModelV2(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(img_path, img_height=64):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    h, w = img.shape

    # Resize keeping aspect ratio
    new_w = int(w * (img_height / h))
    new_w = min(new_w, 512)  # Cap width
    img = cv2.resize(img, (new_w, img_height))

    # Pad to max width
    canvas = np.ones((img_height, 512), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    # Normalize
    canvas = canvas.astype(np.float32) / 255.0

    # Add channel + batch
    canvas = np.expand_dims(canvas, axis=0)  # [1, H, W]
    canvas = np.expand_dims(canvas, axis=0)  # [1, 1, H, W]

    return torch.tensor(canvas)

# =========================
# PREDICT FUNCTION
# =========================
def predict(model, image_paths):
    model.eval()

    for img_path in image_paths:
        img = preprocess_image(img_path).to(DEVICE)

        with torch.no_grad():
            logits1, logits2 = model(img, num_decode_steps=MAX_DECODE_LEN)

        # Use refined output (logits2)
        texts = greedy_decode(logits2)
        text = texts[0]

        print(f"Image: {img_path}")
        print(f"Prediction: {text}")
        print("-" * 60)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = load_model("best_htr_model_v2.pth")
    print("Model loaded successfully\n")

    # 👇 PUT YOUR SAMPLE IMAGES HERE
    sample_images = [
        "samples/presc_49885.png"
    ]

    predict(model, sample_images)
