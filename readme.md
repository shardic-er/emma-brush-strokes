# Image to Path Converter

A Python tool that converts images into normalized path coordinates for artistic rendering. It uses computer vision techniques to detect edges and major features in images, then converts them into a series of coordinate points that could be used for various applications like robotic drawing, CNC machines, or digital art generation.

## Features
- Generates images from text prompts using Pollinations.ai API
- Processes images using edge detection and contour finding
- Converts detected features into normalized coordinates (0.0-1.0 range)
- Saves and displays processing stages (original, edges, contours)
- Filters out small features to focus on major elements

## Requirements
```
numpy==1.24.3
opencv-python==4.8.0.74
Pillow==10.0.0
requests==2.31.0
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the script:
```bash
python main.py
```

The program will:
1. Prompt for an image description
2. Generate and download the image
3. Process it to find major features
4. Display the original image, edge detection, and contours
5. Output normalized coordinate paths for each major feature

## Output
- Images are saved in the `output` directory
- Coordinates are printed as (x, y) pairs in 0.0-1.0 range
- Three windows show processing stages (press any key to close)

## Technical Details
- Uses Canny edge detection with adaptive thresholding
- Implements contour detection and filtering
- Normalizes coordinates for resolution-independent output
- Processes in BGR color space for OpenCV compatibility