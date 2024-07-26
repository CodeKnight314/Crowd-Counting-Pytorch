import cv2
import torchvision.transforms.functional as F
from tqdm import tqdm
from model import CSRNet
import torch
import os
import numpy as np
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def video_inference(input_video: str, model_pth: str, output_path: str):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file '{input_video}' does not exist.")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Error opening video file '{input_video}'.")

    # Loading model and weights
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(model_pth), strict=False)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="[PROCESSING VIDEO]"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(frame_tensor)
            count = prediction.sum().item()

        heatmap = prediction.squeeze(0).cpu().numpy()
        heatmap = cv2.resize(heatmap, (width, height))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
        cv2.putText(overlay, f'Count: {int(count)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(overlay)

    cap.release()
    out.release()
    print("[INFO] Video processing complete.")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input", required=True, type=str, help="input video directory address")
    parser.add_argument("--output", required=True, type=str, help="desired output video directory address")
    parser.add_argument("--model_pth", required=True, type=str, help="model weights for CSRNet")

    args = parser.parse_args()

    video_inference(args.input, args.model_pth, args.output)
