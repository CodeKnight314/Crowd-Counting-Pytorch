import argparse
import os
import torch
from glob import glob
from model import CSRNet
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def img_inference(input_dir: str, output_dir: str, model_pth: str):
    model = CSRNet(load_weights=True).to(device)
    model.load_state_dict(torch.load(model_pth), strict=False)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_paths = glob(os.path.join(input_dir, "*"))
    for img_path in tqdm(img_paths, desc="[IMAGE PROCESSING]"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            count = int(output.detach().cpu().sum().numpy())
            print("Predicted Count:", count)

        temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(temp, cmap='jet', alpha=0.5)
        ax.axis('off')
        ax.set_title(f'Count: {count}')

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print("[INFO] Image processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Folder with input images or single image.")
    parser.add_argument("--output_dir", required=True, type=str, help="Folder to save output files to.")
    parser.add_argument("--model_pth", required=True, type=str, help="Model weights for CSRNet")
    args = parser.parse_args()

    img_inference(args.input_dir, args.output_dir, args.model_pth)
