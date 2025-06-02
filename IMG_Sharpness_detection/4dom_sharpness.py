#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
4DoM Sharpness Estimation Demo
Reference: "Sharpness Estimation for Document and Scene Images"
"""

import cv2
import numpy as np
import argparse
import os

def compute_4dom_sharpness(img_gray, w=2, T=0.05):
    """
    Compute the sharpness score SI based on the 4DoM method proposed in the paper.
    - img_gray: Grayscale image (H x W)
    - w: Difference window size (corresponding to the sum_{i-w}^{i+w} in the paper)
    - T: Threshold for determining whether an edge is sharp
    """
    # 1. Median filtering (adjustable kernel size)
    median_img = cv2.medianBlur(img_gray, 3)
    H, W = median_img.shape

    # 2. Edge detection (quick marking of potential edge pixels) 
    #    Here, for simplicity, we use Sobel or simple differences to obtain "potential edge pixels"
    sobel_x = cv2.Sobel(median_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(median_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_thresh = 0.1 * np.max(magnitude)  # Use 10% of the maximum gradient as the edge threshold
    edge_mask = (magnitude > edge_thresh).astype(np.uint8)

    # 3. Define counters for statistics
    #    #edgePixels_x / #edgePixels_y denote the number of potential edge pixels in horizontal and vertical directions
    #    #sharpPixels_x / #sharpPixels_y denote the number of sharp edge pixels in horizontal and vertical directions
    edgePixels_x = 0
    sharpPixels_x = 0
    edgePixels_y = 0
    sharpPixels_y = 0

    # To avoid index out-of-bound errors when computing the second-order difference,
    # iterate from w to H-w.
    # Horizontal 4DoM: [I(i+2,j) - I(i,j)] - [I(i,j) - I(i-2,j)]
    # Vertical 4DoM: [I(i,j+2) - I(i,j)] - [I(i,j) - I(i,j-2)]
    # The paper uses a fixed distance (i+2, i-2); here a more generic implementation is used.

    # 4. Compute sharpness in the horizontal direction
    for i in range(w, H - w):
        for j in range(W):
            if edge_mask[i, j] > 0:  # If this location is a potential edge
                edgePixels_x += 1
                val_p2 = float(median_img[i + w, j])
                val_p0 = float(median_img[i, j])
                val_m2 = float(median_img[i - w, j])
                diff1 = val_p2 - val_p0
                diff2 = val_p0 - val_m2
                d2 = abs(diff1 - diff2)  # Absolute value of the second-order difference

                # Compute contrast, refer to paper formula (3):
                # contrast = sum(|I(k,j) - I(k-1,j)|), for k in [i-w, i+w]
                # For simplicity, use a local window sum.
                region = median_img[i - w:i + w + 1, j]
                contrast = 0
                # Sum the absolute differences of adjacent pixels along this column
                for r in range(len(region) - 1):
                    contrast += abs(float(region[r + 1]) - float(region[r]))

                # Prevent division by zero
                if contrast < 1e-6:
                    continue
                local_score = d2 / contrast

                # Determine if it is a "sharp" edge based on the threshold
                if local_score > T:
                    sharpPixels_x += 1

    # 5. Compute sharpness in the vertical direction
    for i in range(H):
        for j in range(w, W - w):
            if edge_mask[i, j] > 0:  # If this location is a potential edge
                edgePixels_y += 1
                val_p2 = float(median_img[i, j + w])
                val_p0 = float(median_img[i, j])
                val_m2 = float(median_img[i, j - w])
                diff1 = val_p2 - val_p0
                diff2 = val_p0 - val_m2
                d2 = abs(diff1 - diff2)

                # Compute contrast (along this row)
                region = median_img[i, j - w:j + w + 1]
                contrast = 0
                for c in range(len(region) - 1):
                    contrast += abs(float(region[c + 1]) - float(region[c]))

                if contrast < 1e-6:
                    continue
                local_score = d2 / contrast

                if local_score > T:
                    sharpPixels_y += 1

    # 6. Combine the sharpness scores
    # Rx = #sharpPixels_x / #edgePixels_x
    # Ry = #sharpPixels_y / #edgePixels_y
    # SI = sqrt(Rx^2 + Ry^2)
    # If no edges are detected (edgePixels_x = 0 or edgePixels_y = 0), avoid division by zero.
    Rx = 0.0
    Ry = 0.0
    if edgePixels_x > 0:
        Rx = sharpPixels_x / float(edgePixels_x)
    if edgePixels_y > 0:
        Ry = sharpPixels_y / float(edgePixels_y)

    SI = np.sqrt(Rx**2 + Ry**2)
    return SI, Rx, Ry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to input image", required=True)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold T for deciding sharp edges")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File {args.image} does not exist.")
        return

    # Read the image
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Could not open the image. Check the file path and format.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the sharpness score
    SI, Rx, Ry = compute_4dom_sharpness(gray, w=2, T=args.threshold)

    print(f"Image: {args.image}")
    print(f"Sharpness Score (SI): {SI:.4f}")
    print(f" - Rx (horizontal component): {Rx:.4f}")
    print(f" - Ry (vertical component):   {Ry:.4f}")

if __name__ == "__main__":
    main()
