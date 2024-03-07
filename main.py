"""
`pip install -r requirements.txt` to install all dependencies

This program takes two arrays of evenly spaced x and y values (together defining evenly spaced points on a grid) and
passes them through a set of equations to obtain new x and y values for each point.
This new set of x and y values is passed through the equations again for a given number of iterations, tracing the
path of each starting point over time.

The number of points created is defined by the lattice_size^2 (number of starting points) multiplied by the number of
iterations, which for lattice_size = 100, n_iter = 1000000 (for example) is 10 billion points. Instead of asking our
loyal silicon servants to plot the many billions of points that define our paths, we can create a 2D histogram to bin
the points into a smaller grid, summing the number of times a point is in each bin. The log of this grid is taken to
make the darker values more visible, and it's saved as a 16-bit png.

Use the `generate_random_images` to create lots of images quickly, and then you can see which ones look nice and use
the parameters from the filename in `create_and_plot` to make a more detailed/higher resolution version.
"""


from image_generation_utils import generate_random_images, create_and_plot
from equations import (  # equations from equations.py
    equation_1,
    equation_2,
    equation_6,
    equation_7,
)


########################################################################################################################


### Generate images from random parameters in a range ###

multi_image_config = {
    "param_ranges": [(-1, 1), (-1, 1), (-1, 1)],  # Parameter ranges for equations
    "num_images": 1000,  # Number of images to generate
    "lattice_size": 100,  # Size of the x y grid (higher = denser points).
    "n_iter": 1000,  # Number of times the updated x any y values are passed to the equation (higher = more patterns)
    "bins": 500,  # Histogram bins, and also the image resolution (e.g., 1000 bins is 1000x1000 pixels)
    "save_folder": "test_4",  # Folder to save generated images
    "equations": [equation_1, equation_2, equation_6, equation_7],  # Equations to use for image generation
}

# Uncomment to generate lots of random images
# generate_random_images(**multi_image_config)


########################################################################################################################


### Plot specific equation with specific parameters ###

single_image_config = {
    "params": (-0.22096307, -0.37125962, 0.78463741),  # Parameters for the equation
    "equation": equation_6,  # Equation to use
    "lattice_size": 50,  # Size of the x y grid (higher = denser points).
    "n_iter": 100000,  # Number of times the updated x any y values are passed to the equation (higher = more patterns)
    "bins": 2000,  # Histogram bins, and also the image resolution (e.g., 1000 bins is 1000x1000 pixels)
    "save_folder": "images",  # Folder to save generated images
}

# Uncomment to call create_and_plot to generate and display a single image
create_and_plot(single_image_config)
