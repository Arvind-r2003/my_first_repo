import numpy as np

kernel_conv = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

kernel_corr = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

data_matrix = np.array([[5, 4, 3, 2],
                        [6, 5, 4, 3],
                        [7, 6, 5, 4],
                        [8, 7, 6, 5]])

# Perform convolution
convolution_result = np.zeros_like(data_matrix)

for i in range(data_matrix.shape[0] - 2):
    for j in range(data_matrix.shape[1] - 2):
        window = data_matrix[i:i+3, j:j+3]
        convolution_result[i+1, j+1] = np.sum(window * kernel_conv)

# Perform correlation
correlation_result = np.zeros_like(data_matrix)

for i in range(data_matrix.shape[0] - 2):
    for j in range(data_matrix.shape[1] - 2):
        window = data_matrix[i:i+3, j:j+3]
        correlation_result[i+1, j+1] = np.sum(window * kernel_corr)

print("Convolution Result:")
print(convolution_result)
print("\nCorrelation Result:")
print(correlation_result)
