import argparse
import os
import torch
from glob import glob
from model import CSRNet
import cv2
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def img_inference(input_dir: str, output_dir: str, model_pth: str):
    # Loading model and weights
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_pth), strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    img_paths = glob(os.path.join(input_dir, "*"))
    for img_path in tqdm(img_paths, desc="[IMAGE PROCESSING]"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)
            count = prediction.sum().item()

        heatmap = prediction.squeeze(0).cpu().numpy()
        heatmap = cv2.resize(heatmap, (img.width, img.height))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_cv2, 0.5, heatmap, 0.5, 0)
        cv2.putText(overlay, f'Count: {int(count)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, overlay)

    print("[INFO] Image processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Folder with input images or single image.")
    parser.add_argument("--output_dir", required=True, type=str, help="Folder to save output files to.")
    parser.add_argument("--model_pth", required=True, type=str, help="Model weights for CSRNet")
    args = parser.parse_args()

    img_inference(args.input_dir, args.output_dir, args.model_pth)
