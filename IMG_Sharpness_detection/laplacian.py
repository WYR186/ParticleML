import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def laplacian_variance(image_gray):
    """
    Compute the Laplacian variance of the grayscale image.
    A higher variance typically indicates a sharper (better focused) image.
    """
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    return lap.var()

def tenengrad(image_gray, ksize=3):
    """
    Compute the Tenengrad score of the grayscale image using the Sobel operator.
    This metric reflects the strength of gradients (edges) in the image.
    """
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_square = gx**2 + gy**2
    return np.mean(grad_square)

def combined_focus_score(image_gray, alpha=1.0, beta=1.0):
    """
    Compute a combined focus score by weighting the Laplacian variance and the Tenengrad score.
    The focus score is calculated as:
      score = alpha * (Laplacian Variance) + beta * (Tenengrad Score)
    """
    score_lap = laplacian_variance(image_gray)
    score_ten = tenengrad(image_gray)
    return alpha * score_lap + beta * score_ten

def find_best_focused_image(image_stack, alpha=1.0, beta=1.0, roi=None):
    """
    Iterate through a list (stack) of images to compute each image's focus score,
    and select the best focused image based on the highest score.
    
    Parameters:
      image_stack: a list of images (each can be grayscale or color).
      alpha, beta: weights for the combined focus score.
      roi: a tuple (x, y, w, h) representing a region of interest in the image.
           If None, the full image is used.
    
    Returns:
      best_index: index of the image with the highest focus score.
      best_score: the highest focus score.
      best_img  : the image with the best focus.
      scores    : list of focus scores for each image.
    """
    best_score = -1
    best_index = -1
    best_img = None
    scores = []

    for i, img in enumerate(image_stack):
        # Ensure the image is valid
        if img is None:
            continue

        # Convert to grayscale if the image is in color.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # If a region of interest (ROI) is specified, extract that region.
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # Compute the combined focus score.
        score = combined_focus_score(gray, alpha, beta)
        scores.append(score)

        # Update the best score and image if the current score is higher.
        if score > best_score:
            best_score = score
            best_index = i
            best_img = img

    return best_index, best_score, best_img, scores

def load_images_from_current_directory():
    """
    Load all JPG and PNG images from the current directory.
    
    Returns:
      filenames: list of image file names.
      images   : list of images loaded using cv2.imread.
    """
    # Get all .jpg and .png files in the current folder
    image_files = glob.glob("*.jpg") + glob.glob("*.png")
    images = []
    filenames = []
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
            filenames.append(file)
    return filenames, images

def main():
    # Load images from the current directory.
    filenames, image_stack = load_images_from_current_directory()

    if not image_stack:
        print("No images found in the current directory.")
        return

    # Evaluate all images and select the best focused image.
    best_index, best_score, best_img, scores = find_best_focused_image(image_stack, alpha=1.0, beta=1.0, roi=None)

    # Print the focus score for each image.
    print("Focus Scores for each image:")
    for i, (fname, score) in enumerate(zip(filenames, scores)):
        print(f"Index {i}: {fname} - Focus Score: {score:.2f}")
    print(f"Best focused image: Index {best_index} ({filenames[best_index]}) with Score: {best_score:.2f}")

    # Display all images with their focus scores.
    num_images = len(image_stack)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(image_stack):
        plt.subplot(1, num_images, i+1)
        # Convert from BGR to RGB for correct color display in matplotlib.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{filenames[i]}\nScore: {scores[i]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Display the best focused image separately.
    plt.figure(figsize=(6, 6))
    best_img_rgb = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
    plt.imshow(best_img_rgb)
    plt.title(f"Best Focused Image:\n{filenames[best_index]} with Score: {best_score:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
