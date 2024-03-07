import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from fast_histogram import histogram2d  # pip install fast_histogram
import matplotlib.pyplot as plt


def calculate_paths(lattice_size, params, n_iter, equation, bins):
    """
    Calculates the paths of each point by applying the transformation equation over a specified number of iterations.

    Parameters:
    - lattice_size (int): The size of the grid.
    - params (tuple/list): Parameters for the transformation equation.
    - n_iter (int): Number of iterations to perform.
    - equation (function): The transformation equation to apply to each point.

    Returns:
    - overall_hist (np.array): The aggregated histogram of all points.
    """
    accumulate_every = int(250e6)

    area = np.array([[-1, 1], [-1, 1]])
    x = np.linspace(area[0][0], area[0][1], lattice_size)
    y = np.linspace(area[1][0], area[1][1], lattice_size)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    total_points = n_iter * lattice_size ** 2
    overall_hist = None

    # Pre-allocate arrays if the total points are less than 250 million
    if total_points < accumulate_every:
        x_values = np.zeros(total_points, dtype=np.float16)
        y_values = np.zeros(total_points, dtype=np.float16)
    else:
        x_values = np.zeros(accumulate_every, dtype=np.float16)
        y_values = np.zeros(accumulate_every, dtype=np.float16)

    idx = 0  # Index to track the position in x_values and y_values
    for i in tqdm(range(n_iter), desc="Progress"):
        xx, yy = equation(xx, yy, params)
        flat_xx = xx.flatten().astype(np.float16)
        flat_yy = yy.flatten().astype(np.float16)

        # Determine how many points to write (handling end of array)
        len_to_write = min(len(flat_xx), accumulate_every - idx)

        x_values[idx:idx + len_to_write] = flat_xx[:len_to_write]
        y_values[idx:idx + len_to_write] = flat_yy[:len_to_write]
        idx += len_to_write

        # Check if we need to accumulate histogram and reset
        if idx == accumulate_every or i == n_iter - 1:
            if idx == accumulate_every:
                print("\nAccumulating histogram to save memory...")
            hist = create_histogram(x_values[:idx], y_values[:idx], bins)
            overall_hist = hist if overall_hist is None else overall_hist + hist
            x_values.fill(0)  # Reset the array for new data
            y_values.fill(0)
            idx = 0  # Reset index for the next batch of data

    return overall_hist


def generate_random_images(
        param_ranges, num_images, lattice_size, n_iter, bins, save_folder, equations
):
    """
    Generate a specified number of images using random parameters for the given equations.

    This function creates random images based on the given equations by iterating over a grid of points
    and transforming these points using a randomly selected equation with randomly generated parameters
    in the given ranges.
    Each generated image is saved to a specified folder.

    Parameters:
    - param_ranges (list of tuples): Each tuple specifies the min and max range for the parameters to be used in the equations.
    - num_images (int): The number of images to generate.
    - lattice_size (int): The size of the grid (number of points along one axis).
    - n_iter (int): The number of iterations each point undergoes in the transformation process.
    - bins (int): The number of bins along each axis for the histogram used in creating the image.
    - save_folder (str): The path to the folder where the generated images will be saved.
    - equations (list of functions): A list of functions representing the equations used to transform the points on the grid.

    Returns:
    None: This function does not return any value. It saves the generated images directly to the specified folder.

    Notes:
    - Saves generated images to the filesystem in the specified directory.
    - Prints progress and potential warnings to the standard output.
    """
    params_list = [
        np.random.uniform(low, high, num_images) for (low, high) in param_ranges
    ]

    for i in range(num_images):
        params = [params[i] for params in params_list]
        equation = random.choice(equations)
        plot_specific_equation(params, equation, lattice_size, n_iter, bins, save_folder)


def confirm_execution(number_of_points):
    """
    Prompts the user to confirm execution when the operation involves processing a large number of points.

    This function calculates the number of points to be processed and asks the user for confirmation if this number
    exceeds a certain threshold (1 billion points in this case).

    Parameters:
    - number_of_points (int): The total number of points that will be processed in the operation.

    Returns: - bool: True if the user confirms execution, if the number of points is below the threshold,
    or if the return key is hit without an input; False if the user decides not to proceed with the operation.
    """
    memory_usage_gigabytes = number_of_points * 4 / 1e9  # 4 bytes for each point
    if number_of_points > 1e9:
        response = input(
            f"This will create approximately {number_of_points // 1e9} billion points, which may take a while."
            f"\nAre you sure you want to proceed (Y/n)? ").strip().lower()
        # Check if the response is 'y' or empty (just return key hit)
        if response == 'y' or response == '':
            print("Histograms will be accumulated every 250 million points to save memory.")
            return True
        else:
            return False
    return True



def create_histogram(x_values, y_values, bins):
    """
    Create a histogram image from x and y values.

    This function takes arrays of x and y values, computes a 2D histogram from them,
    and then converts this histogram into a grayscale image. The histogram bins the x
    and y values into a specified number of bins, counts the occurrences in each bin,
    and then the logarithm of each bin count is taken to enhance the visibility of
    lower-density areas. The image is normalized to the full 16-bit range (0-65535).

    Parameters:
    - x_values (np.array): Array of x-values.
    - y_values (np.array): Array of y-values.
    - bins (int): The number of bins to use for the histogram in both x and y directions.

    Returns:
    - img (np.array): The generated image as a 2D numpy array, where each pixel
      intensity corresponds to the logarithm of the count of points in the corresponding
      histogram bin, normalized to the range 0-65535.
    """
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    hist = histogram2d(x_values, y_values, bins=bins, range=[[x_min, x_max], [y_min, y_max]])

    return hist


def create_image(hist, bins):
    pad_size = int(bins * 0.05)
    img = np.zeros((hist.shape[0] + 2 * pad_size, hist.shape[1] + 2 * pad_size), dtype=np.float64)
    img[pad_size:-pad_size, pad_size:-pad_size] = np.log(hist + 1)
    img = (img - img.min()) / (img.max() - img.min()) * 65535
    return img.astype(np.uint16)


def generate_filename(equation_name, params):
    """
    Generate a filename based on the equation name and parameters.

    This function constructs a filename string that incorporates the name of the
    equation function and its parameters. Each parameter is formatted to eight decimal
    places and concatenated using underscores. The filename is intended for saving
    output images, with the parameters included in the name for easy reference.

    Parameters:
    - equation_name (str): The name of the equation, obtained from the
      equation function's `__name__` attribute.
    - params (list or tuple): Numerical parameters used in the equation, which will be
      included in the filename for reference.

    Returns:
    - str: The generated filename, structured as "<equation_name>_<param1>_<param2>_..._<paramN>.png".
    """
    params_str = "_".join([f"{p:.8f}" for p in params])
    filename = f"{equation_name}_{params_str}.png"
    return filename


def plot_specific_equation(params, equation, lattice_size=500, n_iter=100, bins=1000, save_folder="images"):
    """
    Adjusted to directly use the histogram generated in calculate_paths.

    Parameters:
    - params (tuple/list): The parameters to pass to the transformation equation.
    - equation (function): The transformation equation to apply.
    - lattice_size (int): The size of the square grid's side, where the total number of points is lattice_size^2.
    - n_iter (int): The number of iterations to perform the transformation for each point.
    - bins (int): The resolution of the histogram, which defines the resolution of the resulting image.
    - save_folder (str): The directory where the generated image will be saved.

    Returns:
    - img (np.array): The generated image as a 2D numpy array, representing the logarithmic density of point transformations.
    - filename (str): The filename under which the image is saved.
    """
    number_of_points = lattice_size ** 2 * n_iter
    if not confirm_execution(number_of_points):
        print("Operation cancelled.")
        # exit
        exit(0)

    try:
        overall_hist = calculate_paths(lattice_size, params, n_iter, equation, bins)
        if overall_hist is None:
            print("Failed to generate histogram.")
            return

        img = create_image(overall_hist, bins)
        filename = generate_filename(equation.__name__, params)
        image_saver(img, save_folder, filename)
        return img, filename
    except IndexError:
        print("Check you have enough parameters for the chosen equation.")
        exit(-1)
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        exit(-1)


def ensure_directory_exists(directory: str):
    """
    Check if the specified directory exists and create it if it doesn't.

    :param directory: The path of the directory to check and create if necessary.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def image_saver(img, save_folder: str, filename: str):
    """
    Save the image to the specified location, checking the directory exists.

    :param img: The image array to save.
    :param save_folder: The folder where the image should be saved.
    :param filename: The name of the file to save the image as.
    """
    # Ensure the directory exists before trying to save the file
    ensure_directory_exists(save_folder)

    # Construct the full path where the image will be saved
    save_path = os.path.join(save_folder, filename)

    # Save the image using OpenCV
    cv2.imwrite(save_path, cv2.rotate(img.T, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.uint16))
    print(f"Image saved to {save_path}")


def create_and_plot(config):
    """
    Creates and plots an image based on the given configuration.

    Parameters:
    - config (dict): Configuration dictionary containing parameters for the plot_specific_equation function
                     and additional settings for plotting.

    The function attempts to plot a specific equation based on the provided parameters and configuration.
    It then displays the generated image using matplotlib.

    If an error occurs during the image generation or plotting, it will be caught and a message will be printed.
    """
    try:
        img, filename = plot_specific_equation(**config)
        # Plot the image if it was successfully created
        if img is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(img, origin="lower", aspect="auto", cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("Image generation was unsuccessful.")
    except Exception as e:
        print(f"An error occurred: {e}")
