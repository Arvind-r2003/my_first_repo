import numpy as np
import cv2
from PIL import Image
from imgsmooth import generate_gaussian_kernel

def wiener_deconvolution(image, kernel, noise_var):
    image = np.array(image)
    kernel = np.array(kernel)
    output = np.copy(image)
    fft_image = np.fft.fft2(image)
    fft_kernel = np.fft.fft2(kernel, s=image.shape)
    
    # Wiener Deconvolution
    output_fft = np.divide(np.conj(fft_kernel), (np.abs(fft_kernel) ** 2 + noise_var))
    output_fft *= fft_image
    output = np.abs(np.fft.ifft2(output_fft)).astype('uint8')
    
    return Image.fromarray(output)

# Read the image using PIL
image = cv2.imread('images/Lenna.png', cv2.IMREAD_GRAYSCALE)  # Replace 'blurred_image.jpg' with the path to the degraded image

# Create a kernel (for example, a Gaussian kernel)
kernel = generate_gaussian_kernel(kernel_size=5, sigma=1.0)

# Simulate noise variance (you should estimate the actual noise variance)
noise_variance = 1.0

# Apply Wiener deconvolution
restored_image = wiener_deconvolution(image, kernel, noise_variance)

# Display the restored image
restored_image.show()
