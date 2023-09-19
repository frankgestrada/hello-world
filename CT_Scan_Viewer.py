
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon, radon
from skimage import io
from mayavi import mlab
import os
import glob
import numpy as np
from scipy.ndimage import label, center_of_mass


def gather_slice_paths(folder_path):
    """
    Crawl through a specified folder and gather all the slice images (.TIFF files).

    Args:
        folder_path (str): Path to the folder containing the slice images.

    Returns:
        list: List of file paths to the CT slice images.

    Raises:
        Exception: If there are errors in accessing the folder or finding the slice images.
    """
    try:
        # Use glob to find all .TIFF files in the folder
        tiff_files = glob.glob(os.path.join(folder_path, "*.tif"))  # Assumes TIFF files have .tif extension

        if not tiff_files:
            raise Exception("No TIFF files found in the specified folder.")

        return tiff_files

    except Exception as e:
        raise e


def visualize_marbles_sand_can(slice_paths, angle_step=1.0, filter_name='ramp'):
    """
    Visualize the marbles, sand, and can in a CT image in 3D using Mayavi and toggle their visibility.

    Args:
        slice_paths (list): List of file paths to the CT slice images.
        angle_step (float): Angle step in degrees between each CT slice.
        filter_name (str): Filter name for the Radon transform. Options: 'ramp', 'shepp-logan', 'cosine', 'hamming', etc.

    Returns:
        None

    Raises:
        Exception: If there are errors in image reading, reconstruction, or visualization.
    """
    try:
        # Load the CT slice images into an array
        ct_slices = [io.imread(slice_path) for slice_path in slice_paths]

        # Determine the number of projections based on the angle step
        num_projections = int(360.0 / angle_step)

        # Perform Radon transform to get sinogram
        sinogram = radon(np.sum(ct_slices, axis=0), theta=np.linspace(0, 360, num=num_projections), circle=True)

        # Perform inverse Radon transform to reconstruct the CT image
        reconstructed_image = iradon(sinogram, theta=np.linspace(0, 360, num=num_projections), circle=True, filter=filter_name)

        # Create 3D visualization layers for marbles, sand, and can
        mlab.figure(bgcolor=(1, 1, 1))
        marbles_layer = mlab.pipeline.iso_surface(reconstructed_image, colormap='gray', contours=[0.5], opacity=0.7)
        sand_layer = mlab.pipeline.iso_surface(reconstructed_image, colormap='copper', contours=[0.5], opacity=0.7)
        can_layer = mlab.pipeline.iso_surface(reconstructed_image, colormap='bone', contours=[0.5], opacity=0.7)

        # Set visibility of each layer to toggle on/off
        marbles_layer.actor.visibility = True
        sand_layer.actor.visibility = True
        can_layer.actor.visibility = True

        def toggle_visibility(object_name):
            """Toggle the visibility of the specified object."""
            if object_name == "marbles":
                marbles_layer.actor.visibility = not marbles_layer.actor.visibility
            elif object_name == "sand":
                sand_layer.actor.visibility = not sand_layer.actor.visibility
            elif object_name == "can":
                can_layer.actor.visibility = not can_layer.actor.visibility

        # Create a GUI to toggle visibility
        mlab.show()
        mlab.view(azimuth=90, elevation=90, distance='auto')
        mlab.colorbar()
        mlab.title("CT Image Visualization")
        mlab.outline()
        mlab.text(0.01, 0.9, "Toggle Visibility:", width=0.3)
        mlab.text(0.05, 0.85, "Marbles", width=0.1, color=(1, 0, 0), line_width=0.1, name="marbles")
        mlab.text(0.05, 0.80, "Sand", width=0.1, color=(0, 1, 0), line_width=0.1, name="sand")
        mlab.text(0.05, 0.75, "Can", width=0.1, color=(0, 0, 1), line_width=0.1, name="can")

        mlab.show()

    except Exception as e:
        raise e

# Function to save the reconstructed image
def save_reconstructed_image(reconstructed_image, output_path):
    """
    Save the reconstructed CT image to a file.

    Args:
        reconstructed_image (ndarray): The reconstructed CT image.
        output_path (str): File path to save the reconstructed image.

    Returns:
        None

    Raises:
        Exception: If there are errors in saving the image.
    """
    try:
        io.imsave(output_path, reconstructed_image, cmap='gray')
        print(f"Reconstructed image saved to {output_path}")
    except Exception as e:
        raise e

# Example usage to gather slice paths from a folder:
folder_path = "/path/to/your/slice/images/folder"
slice_paths = gather_slice_paths(folder_path)

# Visualize the CT image using the updated function
output_path = "reconstructed_image.tif"
reconstructed_image = visualize_marbles_sand_can(slice_paths, threshold_value=threshold_value)
save_reconstructed_image(reconstructed_image, output_path)


from sklearn.neighbors import NearestNeighbors

def measure_marble_distances(reconstructed_image, num_neighbors=1):
    """
    Measure the distances between the centers of each marble in a CT image.

    Args:
        reconstructed_image (ndarray): The reconstructed CT image with the sand and can layers toggled off.
        num_neighbors (int): The number of nearest neighbors to find for each marble.

    Returns:
        list: List of distances between the centers of each marble and their nearest neighbors.

    Raises:
        Exception: If there are errors in image processing or measurement.
    """
    try:
        # Threshold the reconstructed image to binarize the marbles
        binary_image = reconstructed_image > 0.5  # Adjust the threshold as needed

        # Label the connected components (marbles)
        labeled_image, num_labels = label(binary_image)

        # Initialize a list to store the centers of mass for each marble
        centers = []

        # Calculate the center of mass for each marble
        for label_value in range(1, num_labels + 1):
            com = center_of_mass(binary_image, labeled_image, label_value)
            centers.append(com)

        # Convert the list of centers to a numpy array
        centers = np.array(centers)

        # Find the nearest neighbors for each marble
        nbrs = NearestNeighbors(n_neighbors=num_neighbors+1, algorithm='ball_tree').fit(centers)
        distances, indices = nbrs.kneighbors(centers)

        # Exclude the distance to itself (distance=0) and keep only the distances to the nearest neighbors
        marble_distances = distances[:, 1:]

        return marble_distances

    except Exception as e:
        raise e

# Example usage:
# Assuming you have already reconstructed the CT image with sand and can layers toggled off
# and stored it in the 'reconstructed_image' variable.
distances = measure_marble_distances(reconstructed_image)
print("Distances between marbles:", distances)


import numpy as np
from mayavi import mlab
from scipy.spatial.distance import pdist, squareform


from sklearn.neighbors import NearestNeighbors

def view_marble_3D_matrix(centers, num_neighbors=2):
    """
    Visualize a 3D matrix representing the measured marbles as a connected graph using Mayavi.

    Parameters:
        centers (ndarray): An array containing the center coordinates of each marble.
        num_neighbors (int): The number of nearest neighbors to connect for each marble.

    Raises:
        Exception: If there are any errors during the visualization.
    """
    try:
        # Create a new figure in Mayavi
        mlab.figure(bgcolor=(1, 1, 1))

        # Plot the points representing the marble centers
        mlab.points3d(centers[:, 0], centers[:, 1], centers[:, 2], color=(1, 0, 0), scale_factor=0.5)

        # Find the nearest neighbors for each marble
        nbrs = NearestNeighbors(n_neighbors=num_neighbors+1, algorithm='ball_tree').fit(centers)
        _, indices = nbrs.kneighbors(centers)

        # Plot the edges representing distances between marbles and their nearest neighbors
        for i in range(centers.shape[0]):
            for j in indices[i, 1:]:  # Exclude the first index because it's the marble itself
                x = [centers[i, 0], centers[j, 0]]
                y = [centers[i, 1], centers[j, 1]]
                z = [centers[i, 2], centers[j, 2]]
                mlab.plot3d(x, y, z, color=(0, 0, 1), tube_radius=None, line_width=0.5, scale_factor=0.5)

        # Show the 3D matrix
        mlab.show()

    except Exception as e:
        raise e


# Example usage
if __name__ == "__main__":
    # Simulated centers of marbles for the purpose of demonstration.
    import numpy as np
    import random
    from sklearn.neighbors import NearestNeighbors
    from mayavi import mlab
    from scipy.spatial.distance import pdist, squareform


    def generate_marble_centers_in_jar(jar_radius: float, jar_height: float, marble_radius: float, num_marbles: int) -> np.ndarray:
        """
        Generate the centers of marbles in a cylindrical jar.

        Parameters:
            jar_radius (float): The radius of the cylindrical jar.
            jar_height (float): The height of the jar.
            marble_radius (float): The radius of the marbles.
            num_marbles (int): The number of marbles in the jar.

        Returns:
            np.ndarray: An array containing the 3D coordinates of the centers of the marbles.

        Raises:
            Exception: If parameters are invalid or if unable to fit the specified number of marbles.
        """
        try:
            if jar_radius <= 0 or jar_height <= 0 or marble_radius <= 0 or num_marbles <= 0:
                raise ValueError("Parameters must be positive numbers.")

            marble_centers = []

            max_attempts = 1000
            for _ in range(num_marbles):
                for _ in range(max_attempts):
                    x = random.uniform(-jar_radius + marble_radius, jar_radius - marble_radius)
                    y = random.uniform(-jar_radius + marble_radius, jar_radius - marble_radius)
                    if x ** 2 + y ** 2 <= (jar_radius - marble_radius) ** 2:
                        z = random.uniform(0 + marble_radius, jar_height - marble_radius)

                        overlap = any(np.linalg.norm(np.array([x, y, z]) - np.array(center)) < 2 * marble_radius for center in marble_centers)

                        if not overlap:
                            marble_centers.append([x, y, z])
                            break
                else:
                    raise Exception("Could not fit the specified number of marbles.")

            return np.array(marble_centers)

        except Exception as e:
            raise e

    jar_radius = 25
    jar_height = 100
    marble_radius = 1
    num_marbles = 300

    try:
        centers = generate_marble_centers_in_jar(jar_radius, jar_height, marble_radius, num_marbles)
        print("Marble centers:\n", centers)
        view_marble_3D_matrix(centers, 2)
    except Exception as e:
        print(f"An error occurred: {e}")