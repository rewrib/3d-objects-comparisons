import os

import matplotlib.pyplot as plt
import numpy as np

# Constants for paths
SOURCE1_PATH = "D:\\Gesicht"
SOURCE2_PATH = "D:\\auto3d\\Vergleich-Output\\phase2\\aligned_landmarks"


def compute_point_cloud_similarity(matrix1, matrix2):
    """
    Compute the Euclidean distance between two matrices.

    Parameters:
    - matrix1 (numpy.ndarray): First matrix.
    - matrix2 (numpy.ndarray): Second matrix.

    Returns:
    - distance (float): Euclidean distance between matrices.
    """
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    # Check if the matrices have the same shape
    if array1.shape != array2.shape:
        raise ValueError(
            "Matrices must have the same shape for Euclidean distance calculation."
        )

    # Calculate the Euclidean distance
    distance = np.linalg.norm(array1 - array2)
    return distance


def visualize_similarity_matrix(similarity_matrix, source1_files, source2_files):
    """
    Visualize the similarity matrix between two sources through a heatmap.

    Parameters:
    - similarity_matrix (numpy.ndarray): Matrix of similarity scores.
    - source1_files (list): List of filenames for source 1.
    - source2_files (list): List of filenames for source 2.
    """

    # Create a subplot for the visualization
    fig, ax = plt.subplots()

    # Display the similarity matrix as an image with a colormap
    im = ax.imshow(similarity_matrix, cmap="viridis_r", interpolation="nearest")

    # Set tick labels and show filenames on the x and y axes
    ax.set_xticks(np.arange(len(source2_files)))
    ax.set_yticks(np.arange(len(source1_files)))
    ax.set_xticklabels(source2_files, rotation=45, ha="right")
    ax.set_yticklabels(source1_files)

    # Add a color bar to the plot
    plt.colorbar(im, label="Similarity Score")

    # Set axis labels and invert the x-axis
    plt.xlabel("Auto3dgm")
    plt.ylabel("Deep-MVLM")
    plt.gca().invert_xaxis()
    plt.title("Point Cloud Similarity Matrix")

    # Add text annotations for each cell
    for i in range(len(source1_files)):
        for j in range(len(source2_files)):
            text = ax.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
            )

    plt.tight_layout()
    plt.show()


def convert_from_face_algorithm(file_path):
    """
    Convert landmarks from Deep-MVLM output file into a numpy matrix

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - numpy.ndarray: Point cloud data.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    return np.array([list(map(float, line.split())) for line in lines])


def convert_from_auto3d(file_path):
    """
    Convert landmarks from auto3dgm output file into a numpy matrix

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - numpy.ndarray: Point cloud data.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract x, y, z columns into a matrix
    return np.array(
        [
            list(map(float, line.split(",")[1:4]))
            for line in lines
            if not line.startswith("#")
        ]
    )


def compute_similarity_matrix(source1_files, source2_files, source1_path, source2_path):
    """
    Compute the similarity matrix between point clouds from two sources.

    Parameters:
    - source1_files (list): List of filenames for source 1.
    - source2_files (list): List of filenames for source 2.
    - source1_path (str): Path to source 1 files.
    - source2_path (str): Path to source 2 files.

    Returns:
    - similarity_matrix (numpy.ndarray): Matrix of similarity scores.
    """
    num_source1_files = len(source1_files)
    num_source2_files = len(source2_files)

    similarity_matrix = np.zeros((num_source1_files, num_source2_files))

    # Iterate over each pair of files and calculate similarity
    for i in range(num_source1_files):
        for j in range(num_source2_files):
            file1_path = os.path.join(source1_path, source1_files[i])
            file2_path = os.path.join(source2_path, source2_files[j])

            cloud1 = convert_from_face_algorithm(file1_path)
            cloud2 = convert_from_auto3d(file2_path)

            similarity_score = compute_point_cloud_similarity(cloud1, cloud2)
            similarity_matrix[i, j] = similarity_score

    return similarity_matrix


def main():
    source1_files = [f for f in os.listdir(SOURCE1_PATH) if f.endswith(".txt")]
    source2_files = [f for f in os.listdir(SOURCE2_PATH) if f.endswith(".fcsv")]

    similarity_matrix = compute_similarity_matrix(
        source1_files, source2_files, SOURCE1_PATH, SOURCE2_PATH
    )

    visualize_similarity_matrix(similarity_matrix, source1_files, source2_files)


if __name__ == "__main__":
    main()
