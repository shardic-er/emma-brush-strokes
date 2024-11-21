import cv2
import numpy as np
from typing import List, Tuple
import requests
from io import BytesIO
from PIL import Image
import urllib.parse
import os
import threading


def ensure_output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')


def get_image_from_pollinations(prompt: str) -> np.ndarray:
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
    print("Generating image...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    ensure_output_dir()
    img.save('output/original.png')
    print("Image saved!")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def process_image(image: np.ndarray, min_contour_area: float = 200) -> List[List[Tuple[float, float]]]:
    print("Processing image...")
    # Convert to grayscale and apply moderate blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Use Canny with automatic threshold detection
    median = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blurred, lower, upper)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save processing result
    cv2.imwrite('output/edges.png', dilated)

    height, width = image.shape[:2]
    paths = []

    # Create visualization
    contour_img = np.zeros_like(image)
    filtered_contours = []

    for contour in contours:
        # Filter small contours but keep more than before
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Smooth the contour with less aggressive parameters
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to normalized coordinates
        normalized_points = [(float(x) / width, float(y) / height)
                             for [[x, y]] in approx]

        paths.append(normalized_points)
        filtered_contours.append(approx)

    # Draw filtered contours
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imwrite('output/contours.png', contour_img)

    return paths


def show_images():
    threading.Thread(target=lambda: [
        cv2.imshow('Original Image', cv2.imread('output/original.png')),
        cv2.imshow('Processed Edges', cv2.imread('output/edges.png')),
        cv2.imshow('Detected Contours', cv2.imread('output/contours.png')),
        cv2.waitKey(0),
        cv2.destroyAllWindows()
    ], daemon=True).start()


def main(prompt: str) -> List[List[Tuple[float, float]]]:
    try:
        image = get_image_from_pollinations(prompt)
        paths = process_image(image)
        print("Processing complete!")
        print(f"Found {len(paths)} major features")
        show_images()
        return paths
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []


if __name__ == "__main__":
    prompt = input("Enter image prompt: ")
    paths = main(prompt)
    if paths:
        print("\nPath points for major features:")
        for i, path in enumerate(paths):
            print(f"\nFeature {i + 1} ({len(path)} points):")
            for x, y in path[:3]:
                print(f"  ({x:.3f}, {y:.3f})")
            if len(path) > 6:
                print("  ...")
            for x, y in path[-3:]:
                print(f"  ({x:.3f}, {y:.3f})")