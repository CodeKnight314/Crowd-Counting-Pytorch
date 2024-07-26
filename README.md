# Crowd-Counting-Pytorch

## Overview
This repository is built for educational purposes on CSRNet in crowd counting, a frequented practice by others over the past half-decade. This repository's attempt at novelty focuses on the application of the model with crowd behavior visualization. Alongside the commonly implemented CSRNet model, image and video inference scripts are developed for public usage. However, video script currently does not support real-time inference through camera.

It's well demonstrated that CSRNet can density based regression which predicts the number of people in the crowd without using bounding boxes. However, since the model is directly predicting the crowd density of a crowd, the model is prone to higher error rates. 

## Usage
- For image inference, run the following code snippet.

```python
python img_inference.py --input_dir path/to/image_folder --output_dir path/to/output_folder --model_pth path/to/model_pth
```

- For video inference, run the following code snippet.

```python
python video_inference.py --input path/to/video.mp4 --output path/to/output.mp4 --model_pth path/to/model_pth
```