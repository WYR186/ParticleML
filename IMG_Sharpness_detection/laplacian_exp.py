import cv2  # OpenCV library for image processing tasks
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting images and results
import glob  # Glob for file pattern matching in the filesystem

def laplacian_variance(image_gray):
    """
    Compute the Laplacian variance of the grayscale image.
    A higher variance typically indicates a sharper (better focused) image.
    """
    # Apply the Laplacian operator to highlight areas of rapid intensity change (edges)
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    # Calculate and return the statistical variance of the Laplacian image
    return lap.var()

def tenengrad(image_gray, ksize=3):
    """
    Compute the Tenengrad score of the grayscale image using the Sobel operator.
    This metric reflects the strength of gradients (edges) in the image.
    """
    # Compute horizontal gradients (derivative in x-direction)
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    # Compute vertical gradients (derivative in y-direction)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Combine squared gradients to get gradient magnitude squared
    grad_square = gx**2 + gy**2
    # Return the mean gradient magnitude squared as the focus measure
    return np.mean(grad_square)

def combined_focus_score(image_gray, alpha=1.0, beta=1.0):
    """
    Compute a combined focus score by weighting the Laplacian variance and the Tenengrad score.
    Formula:
      score = alpha * (Laplacian Variance) + beta * (Tenengrad Score)
    """
    # Calculate individual focus metrics
    score_lap = laplacian_variance(image_gray)
    score_ten = tenengrad(image_gray)
    # Combine them with given weights to produce a single focus score
    return alpha * score_lap + beta * score_ten

def find_best_focused_image(image_stack, alpha=1.0, beta=1.0, roi=None):
    """
    Iterate through a list (stack) of images to compute each image's focus score,
    and select the best focused image based on the highest score.

    Parameters:
      image_stack: list of images (grayscale or BGR color).
      alpha, beta: weights for the combined focus score.
      roi: (x, y, w, h) tuple defining a region of interest. If None, use full image.

    Returns:
      best_index: index of the image with the highest combined focus score.
      best_score: that highest focus score.
      best_img  : the image corresponding to best_score.
      scores    : list of all computed focus scores.
    """
    best_score = -1  # Initialize with a very low score
    best_index = -1  # Placeholder for the index of the best image
    best_img = None  # Placeholder for the best image itself
    scores = []      # To store each image's focus score

    for i, img in enumerate(image_stack):
        if img is None:
            # Skip invalid or unreadable images
            continue

        # Convert color images to grayscale for focus computation
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img  # Already grayscale

        # Crop to region of interest if provided
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # Compute the combined focus score for the current image
        score = combined_focus_score(gray, alpha, beta)
        scores.append(score)

        # Update best score and image when a higher score is found
        if score > best_score:
            best_score = score
            best_index = i
            best_img = img

    return best_index, best_score, best_img, scores

def load_images_from_current_directory():
    """
    Load all JPG and PNG images from the current directory.

    Returns:
      filenames: list of image filenames.
      images   : list of images read via cv2.imread.
    """
    # Find all files ending in .jpg or .png
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
    # Load images (and their filenames) from the working directory
    filenames, image_stack = load_images_from_current_directory()

    # If no images are found, inform the user and exit
    if not image_stack:
        print("No images found in the current directory.")
        return

    # Identify the best-focused image among the loaded set
    best_index, best_score, best_img, scores = find_best_focused_image(
        image_stack, alpha=1.0, beta=1.0, roi=None
    )

    # Print focus scores for each image
    print("Focus Scores for each image:")
    for i, (fname, score) in enumerate(zip(filenames, scores)):
        print(f"Index {i}: {fname} - Focus Score: {score:.2f}")
    print(f"Best focused image: Index {best_index} "
          f"({filenames[best_index]}) with Score: {best_score:.2f}")

    # Display all images side by side with their respective focus scores
    num_images = len(image_stack)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(image_stack):
        plt.subplot(1, num_images, i+1)
        # OpenCV uses BGR; convert to RGB for correct Matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{filenames[i]}\nScore: {scores[i]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Show the single best-focused image in a larger view
    plt.figure(figsize=(6, 6))
    best_img_rgb = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
    plt.imshow(best_img_rgb)
    plt.title(f"Best Focused Image:\n{filenames[best_index]} "
              f"with Score: {best_score:.2f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
