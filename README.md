## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenCV
- tqdm
- fast-histogram

## Installation Instructions

### Clone the Repository

First, clone the repository to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/tristanmanchester/some_nice_python_art
```
Then, install the dependencies:
```bash
cd some_nice_python_art
pip install -r requirements.txt
```


## Usage

The program consists of the following main files:

### `main.py`

This is the entry point of the program. It contains two main sections:

1. **Generate images from random parameters in a range**
   - Uncomment the `generate_random_images` call to create a batch of images with randomly generated parameters.
   - Adjust the configuration parameters (`param_ranges`, `num_images`, `lattice_size`, `n_iter`, `bins`, `save_folder`, `equations`) as desired.

2. **Plot a specific equation with specific parameters**
   - Uncomment the `create_and_plot` call to generate and display a single image with specific parameters.
   - Modify the `single_image_config` dictionary with the desired parameters (`params`, `equation`, `lattice_size`, `n_iter`, `bins`, `save_folder`).

### `image_generation_utils.py`

This module contains the core functions for generating and saving images:

- `calculate_paths`: Calculates the paths of each point by applying the transformation equation over a specified number of iterations.
- `generate_random_images`: Generates a specified number of images using random parameters for the given equations.
- `confirm_execution`: Prompts the user to confirm execution when the operation involves processing a large number of points.
- `create_histogram`: Creates a histogram from x and y values.
- `create_image`: Converts a histogram into a grayscale image.
- `generate_filename`: Generates a filename based on the equation name and parameters.
- `plot_specific_equation`: Generates an image for a specific equation and set of parameters.
- `ensure_directory_exists`: Checks if a directory exists and creates it if necessary.
- `image_saver`: Saves the generated image to the specified location.
- `create_and_plot`: Creates and plots an image based on the given configuration.

### `equations.py`

This file contains the mathematical equations used for transforming the points on the grid. You can add your own equations to this file and import them into `main.py`.

## Examples

Here are some examples of the images you can generate with this program:

![Example Image 1](example_images/equation_6_-0.22096307_-0.37125962_0.78463741.png)
![Example Image 2](example_images/equation_7_-0.22096307_-0.37125962_0.78463741.png)
