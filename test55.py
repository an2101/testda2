import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "img_size": 224,
    "seg_threshold": 0.5,
    "dropout_rate": 0.5,
}

# ================= TRANSFORM =================
def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ================= MODEL =================
class MobileNetV2Classifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        bb = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = bb.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.classifier(x)
        return x.squeeze(1)

class DoubleConv(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,padding=1,bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class LungUNet(nn.Module):
    def __init__(self, features=[64,128,256,512]):
        super().__init__()
        self.downs=nn.ModuleList(); self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(2)

        ch=3
        for f in features:
            self.downs.append(DoubleConv(ch,f)); ch=f

        self.bottleneck=DoubleConv(features[-1],features[-1]*2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2,f,2,2))
            self.ups.append(DoubleConv(f*2,f))

        self.final=nn.Conv2d(features[0],1,1)

    def forward(self,x):
        skips=[]
        for d in self.downs:
            x=d(x); skips.append(x); x=self.pool(x)

        x=self.bottleneck(x); skips=skips[::-1]

        for i in range(0,len(self.ups),2):
            x=self.ups[i](x)
            s=skips[i//2]
            if x.shape!=s.shape:
                x=F.interpolate(x,size=s.shape[2:])
            x=torch.cat([s,x],dim=1)
            x=self.ups[i+1](x)

        return self.final(x)

# ================= LOAD MODEL =================
@st.cache_resource

import gdown
import os

MODEL_DIR = "/tmp/model"   # 🔥 dùng /tmp cho cloud (an toàn)

def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

@st.cache_resource
def load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 🔥 THAY ID thật của bạn vào đây
    UNET_ID = "1KoBmhwFMZJV6GVpabjgcY2aFGU-kcm03"
    MOBILENET_ID = "1g_DYDpEqnpzfCa3UIAE0ifJHVAHZyIjW"

    unet_path = os.path.join(MODEL_DIR, "lung_unet.pth")
    mobile_path = os.path.join(MODEL_DIR, "mobilenet.pth")

    # 🔥 download nếu chưa có
    download_model(UNET_ID, unet_path)
    download_model(MOBILENET_ID, mobile_path)

    # load model
    lung_unet = LungUNet().to(DEVICE)
    mobilenet = MobileNetV2Classifier(CFG["dropout_rate"]).to(DEVICE)

    lung_unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
    mobilenet.load_state_dict(torch.load(mobile_path, map_location=DEVICE))

    lung_unet.eval()
    mobilenet.eval()

    return lung_unet, mobilenet



lung_unet, mobilenet = load_models()

# ================= UI =================
st.title("🫁 Pneumonia Detection (UNet + MobileNetV2)")

uploaded_file = st.file_uploader("Upload ảnh X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Ảnh gốc", use_container_width=True)

    # ===== UNET =====
    img_resize = cv2.resize(img_rgb, (CFG["img_size"], CFG["img_size"]))
    tfm = get_transforms(CFG["img_size"])
    img_tensor = tfm(image=img_resize)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = torch.sigmoid(lung_unet(img_tensor))
        mask = (mask > CFG["seg_threshold"]).float()

    mask_np = mask.squeeze().cpu().numpy()

    st.image(mask_np, caption="UNet Mask", use_container_width=True)

    # ===== BBOX =====
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)

    if not rows.any():
        crop_resize = cv2.resize(img_rgb, (224,224))
    else:
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        pad = 0.1
        H, W = mask_np.shape

        pr = int((r1 - r0) * pad)
        pc = int((c1 - c0) * pad)

        r0, r1 = max(0, r0 - pr), min(H, r1 + pr)
        c0, c1 = max(0, c0 - pc), min(W, c1 + pc)

        crop = img_resize[r0:r1, c0:c1]
        crop_resize = cv2.resize(crop, (224,224))

    st.image(crop_resize, caption="Cropped Lung", use_container_width=True)

    # ===== CLASSIFICATION =====
    clf_tfm = get_transforms(224)
    x = clf_tfm(image=crop_resize)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = mobilenet(x)
        prob = torch.sigmoid(output).item()

    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    st.subheader("Kết quả")
    st.write(f"Prediction: **{label}**")
    st.write(f"Probability: **{prob:.4f}**")

