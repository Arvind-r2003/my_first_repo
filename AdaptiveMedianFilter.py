import numpy as np

def adaptive_median_filter(image, window_size, threshold):
    
    # Create a padded image to handle edge cases.
    padded_image = np.pad(image, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)), 'constant')

    # Create an output image.
    filtered_image = np.zeros_like(image)

    # Iterate over the image.
    for i in range(window_size // 2, image.shape[0] - window_size // 2):
        for j in range(window_size // 2, image.shape[1] - window_size // 2):
            # Get the window of pixels around the current pixel.
            window = padded_image[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1]

            # Calculate the median of the window.
            median = np.median(window)

            # Calculate the absolute differences between the current pixel and the median.
            abs_diffs = np.abs(padded_image[i, j] - median)

            # If the absolute difference is greater than the threshold, then the pixel is noise.
            if abs_diffs > threshold:
                # Replace the current pixel with the median.
                filtered_image[i - window_size // 2, j - window_size // 2] = median
            else:
                # Leave the current pixel unchanged.
                filtered_image[i - window_size // 2, j - window_size // 2] = padded_image[i, j]

    return filtered_image

# Example usage:
image = np.array([[10, 15, 20, 15, 10],
                         [15, 20, 25, 20, 15],
                         [20, 25, 30, 25, 20],
                         [15, 20, 25, 20, 15],
                         [10, 15, 20, 15, 10]])

# Apply the adaptive median filter with a window size of 3 and a threshold of 10.
filtered_image = adaptive_median_filter(image, 3, 10)

# Print the filtered image.
print(filtered_image)
