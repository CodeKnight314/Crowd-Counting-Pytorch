# Crowd-Counting-Pytorch

## Overview
Overview
This repository is built for educational purposes on CSRNet in crowd counting, a frequent practice by others over the past half-decade. Crowd counting is a crucial task in computer vision, aimed at estimating the number of individuals in an image or video. It has a wide range of applications, from public safety and event management to retail analytics and urban planning. 

This repository includes:
- CSRNet Model Implementation: A PyTorch implementation of the CSRNet model.
- Image Inference Script: A script to perform inference on a folder of images, generating crowd density heatmaps and overlaying them on the original images.
- Video Inference Script: A script to perform inference on a video file, generating crowd density heatmaps for each frame and overlaying them on the original frames.

These scripts provide a simple and effective way to apply CSRNet to real-world data, enabling users to quickly visualize and analyze crowd density in their images and videos.

## Usage
- For image inference, run the following code snippet.

```python
python img_inference.py --input_dir path/to/image_folder --output_dir path/to/output_folder --model_pth path/to/model_pth
```

- For video inference, run the following code snippet.

```python
python video_inference.py --input path/to/video.mp4 --output path/to/output.mp4 --model_pth path/to/model_pth
```

# Model Weights
The pre-trained model weights for CSRNet can be downloaded directly from this repository. Make sure to place the downloaded weights file in the appropriate directory or specify the correct path when running inference.