from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import segmentation_models_pytorch as smp
import io
import os
import gdown

app = FastAPI(title="Seed Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Blwrc -GA', 'Bwr-2', 'Jyothi', 'Kau Manu ratna(km-1)', 'Menu verna', 'Pour nami (p-1)', 'Sreyas', 'Uma -1']
NUM_CLASSES = len(CLASS_NAMES)
CONF_THRESHOLD = 60
SEG_MODEL_PATH = "seed_unet_best.pt"
CLS_MODEL_PATH = "vit_se_epoch_10.pth"

# ─── Google Drive File IDs ─────────────────────────────────
# Replace these with your actual Google Drive file IDs
SEG_MODEL_GDRIVE_ID = "YOUR_UNET_FILE_ID"
CLS_MODEL_GDRIVE_ID = "YOUR_VIT_FILE_ID"

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (255, 165, 0)
]
label_to_color = {CLASS_NAMES[i]: CLASS_COLORS[i] for i in range(NUM_CLASSES)}

# ─── Download Models if Not Present ──────────────────────
def download_models():
    if not os.path.exists(SEG_MODEL_PATH):
        print("Downloading UNet model...")
        gdown.download(f"https://drive.google.com/uc?id={SEG_MODEL_GDRIVE_ID}", SEG_MODEL_PATH, quiet=False)
        print("UNet model downloaded.")

    if not os.path.exists(CLS_MODEL_PATH):
        print("Downloading ViT-CBAM model...")
        gdown.download(f"https://drive.google.com/uc?id={CLS_MODEL_GDRIVE_ID}", CLS_MODEL_PATH, quiet=False)
        print("ViT-CBAM model downloaded.")

download_models()

# ─── Model Definitions ────────────────────────────────────
class CBAMBlock(nn.Module):
    def __init__(self, embed_dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction)
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        channel_att = self.sigmoid(avg_out + max_out).unsqueeze(1)
        x = x * channel_att
        avg_pool = x.mean(dim=2, keepdim=True)
        max_pool, _ = x.max(dim=2, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool], dim=2)
        spatial = spatial.permute(0, 2, 1)
        spatial_att = self.sigmoid(self.conv(spatial))
        spatial_att = spatial_att.permute(0, 2, 1)
        x = x * spatial_att
        return x

class ViT_CBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        embed_dim = self.vit.hidden_dim
        self.cbam = CBAMBlock(embed_dim)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.cbam(x)
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        cls = x[:, 0]
        return self.vit.heads(cls)

# ─── Load Models ──────────────────────────────────────────
seg_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
seg_model.to(DEVICE).eval()

class_model = ViT_CBAM(NUM_CLASSES)
class_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
class_model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Pipeline Functions ───────────────────────────────────
def predict_mask(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (768, 768))
    R, G, B = r[:,:,0], r[:,:,1], r[:,:,2]
    T = 20
    grey_mask = (np.abs(R-G) < T) & (np.abs(G-B) < T) & (np.abs(R-B) < T)
    gray = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    bright_mask = gray > 100
    bg_mask = grey_mask & bright_mask
    r[bg_mask] = [255, 255, 255]
    x = r.transpose(2,0,1)[None] / 255.0
    x = torch.tensor(x).float().to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(seg_model(x))[0,0].cpu().numpy()
    p = (p > 0.6).astype(np.uint8)
    kernel_big = np.ones((7,7), np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel_big)
    kernel = np.ones((3,3), np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, kernel)
    p = cv2.resize(p, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return p * 255

def split_instances_erosion(mask, min_area=2000):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
    outs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comp = cv2.dilate(comp, kernel, iterations=3)
        outs.append(comp)
    if len(outs) == 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                outs.append((labels == i).astype(np.uint8)*255)
    return outs

def extract_instances_white(orig, masks):
    outs = []
    for m in masks:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 800:
                continue
            comp = (labels == i)
            ys, xs = np.where(comp)
            if len(xs) == 0:
                continue
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop_img = orig[y1:y2+1, x1:x2+1]
            crop_mask = comp[y1:y2+1, x1:x2+1]
            gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            texture = np.var(gray_crop[crop_mask])
            if texture < 50:
                continue
            canvas = np.ones_like(crop_img, dtype=np.uint8) * 255
            canvas[crop_mask] = crop_img[crop_mask]
            outs.append((canvas, crop_mask.astype(np.uint8), (x1, y1, x2-x1+1, y2-y1+1)))
    return outs

def pad_to_square(img_np):
    h, w = img_np.shape[:2]
    diff = abs(h - w)
    pad1, pad2 = diff // 2, diff - diff // 2
    if h < w:
        img_np = cv2.copyMakeBorder(img_np, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        img_np = cv2.copyMakeBorder(img_np, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[255,255,255])
    return img_np

def predict_image(img_np):
    img_np = pad_to_square(img_np)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = class_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    confidence = conf.item() * 100
    if confidence < CONF_THRESHOLD:
        return "Unknown", confidence
    return CLASS_NAMES[pred.item()], confidence

def overlay_masks_on_image(orig, instances, alpha=0.5):
    overlay = orig.copy()
    for crop, mask, (x, y, w, h) in instances:
        pred, conf = predict_image(crop)
        if pred == "Unknown":
            color = np.array([128, 128, 128], dtype=np.uint8)
            label_text = "Unknown"
        else:
            color = np.array(label_to_color[pred], dtype=np.uint8)
            label_text = pred
        region = overlay[y:y+h, x:x+w]
        colored = np.zeros_like(region)
        colored[:] = color
        mask_3 = mask[:, :, None] > 0
        region = np.where(
            mask_3,
            (alpha * colored + (1 - alpha) * region).astype(np.uint8),
            region
        )
        overlay[y:y+h, x:x+w] = region
        text_x = x
        text_y = y - 5 if y - 5 > 10 else y + 15
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (text_x, text_y - th - 4), (text_x + tw + 4, text_y + 2), (0, 0, 0), -1)
        cv2.putText(overlay, label_text, (text_x + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay

# ─── Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    mask = predict_mask(img_bgr)
    instances = split_instances_erosion(mask)
    crops = extract_instances_white(img_bgr, instances)

    result = overlay_masks_on_image(img_bgr, crops, alpha=0.5)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")