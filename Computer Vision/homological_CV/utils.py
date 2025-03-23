import numpy as np
from matplotlib import pyplot as plt


def plot_transformed_points(original, transformed, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(original[:, 0], original[:, 1], color='blue', label='Original')
    plt.scatter(transformed[:, 0], transformed[:, 1], color='red', label='Transformed')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_transformation(img, matrix):
    output = np.zeros_like(img)
    # Get the center coordinates
    cx, cy = img.shape[0] // 2, img.shape[1] // 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                # Move origin to the center of the image before applying the transformation
                original_point = np.array([i - cx, j - cy, 1])
                transformed_point = matrix @ original_point
                # Move origin back to the top-left corner
                x, y = int(round(transformed_point[0] + cx)), int(round(transformed_point[1] + cy))
                if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                    output[x, y] = 255
    return output


def apply_transformation_vectorized(img, matrix):
    rows, cols = img.shape
    # Create a grid of coordinates
    y_coords, x_coords = np.indices((rows, cols))
    # Stack to create homogeneous coordinates (3, rows, cols)
    coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)])
    # Reshape for matrix multiplication (3, rows*cols)
    coords = coords.reshape(3, -1)
    # Apply the transformation matrix
    transformed_coords = matrix @ coords
    # Normalize homogeneous coordinates
    transformed_coords //= transformed_coords[-1, :]
    # Round and convert to integer indices
    x_transformed, y_transformed = transformed_coords[:2, :].astype(int)
    # Clip to ensure coordinates remain within image bounds
    x_transformed = np.clip(x_transformed, 0, cols - 1)
    y_transformed = np.clip(y_transformed, 0, rows - 1)
    # Map coordinates back to a grid and reshape to the image size
    output_img = img[y_transformed, x_transformed].reshape(rows, cols)
    return output_img
