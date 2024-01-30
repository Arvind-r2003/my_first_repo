import numpy as np
import cv2
from PIL import Image

def gaussian_blur(image, kernel_size, sigma):
    image = np.array(image)
    height, width = image.shape
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    output = np.zeros_like(image)
    padding = kernel_size // 2

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            region = image[i - padding: i + padding + 1, j - padding: j + padding + 1]
            weighted_sum = np.sum(region * kernel)
            output[i, j] = weighted_sum

    return Image.fromarray(output)

def generate_gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(- ((x - (kernel_size // 2))**2 + (y - (kernel_size // 2))**2) / (2 * sigma**2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)

# Read the image using PIL
image = cv2.imread('images/Lenna.png', cv2.IMREAD_GRAYSCALE)
  # Replace 'your_image.jpg' with the path to your image

# Apply Gaussian blur
blurred_image = gaussian_blur(image, kernel_size=5, sigma=1.0)

# Display the blurred image

blurred_image.show()
