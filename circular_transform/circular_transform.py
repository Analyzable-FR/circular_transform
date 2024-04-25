"""
MIT License

Copyright (c) 2024 Analyzable | Benjamin Gallois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
import shapely


class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        affines = np.array(
            [self.affines[i].params for i in range(
                len(self._tesselation.simplices))]
        )[simplex]

        pts = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1

        return result


def transformation_function_linear(radius, dr, desired_radius):
    """
    Linear transformation function for modifying a contour into a circle.

    Parameters:
    - radius (array-like): Original radius values.
    - dr (float): Distance between the contour an the desired circle.
    - desired_radius (float): Desired circle radius after transformation.

    Returns:
    - numpy.ndarray: Transformed radius values.
    """
    return (desired_radius / (desired_radius + dr)) * np.asarray(radius)


def get_transformation_grid(delta_r, number_radius, number_theta, max_radius, center,
                            desired_radius, transformation_function=transformation_function_linear, verbose=False):
    """
    Generate original and transformed grids based to transform a contour in a circle.

    Parameters:
    - delta_r (list): List of radius differences.
    - number_radius (int): Number of radius values.
    - number_theta (int): Number of theta values.
    - max_radius (float): Maximum radius value.
    - center (tuple): Center coordinates of the grid.
    - desired_radius (float): Desired circle radius after transformation.
    - transformation_function (function): Function to compute radius transformation.
    - verbose (bool): If True, plot the grids.

    Returns:
    - list: List containing original and transformed grids.
    """
    r = np.linspace(0, max_radius, number_radius)
    theta = np.linspace(0, 2 * np.pi, number_theta)

    # Original grid
    X = []
    Y = []
    for j in theta:
        X.extend(r * np.cos(j) + center[0])
        Y.extend(r * np.sin(j) + center[1])

    # Transformed grid
    X_ = []
    Y_ = []
    for l, i in enumerate(theta):
        r_ = transformation_function(r, delta_r[l], desired_radius)
        X_.extend(r_ * np.cos(i) + center[0])
        Y_.extend(r_ * np.sin(i) + center[1])

    if verbose:
        plt.scatter(X, Y, color="blue", label="Original grid")
        plt.scatter(X_, Y_, color="red", label="Transformed grid")
        plt.xlim(0, max_radius)
        plt.ylim(0, max_radius)
        plt.legend()

    return [[X, Y], [X_, Y_]]


def compute_radius_difference(
        object_contour, number_radius, number_theta, max_radius, center, desired_radius):
    """
    Compute the difference in radius between the object contour and the circular shape.

    Parameters:
    - object_contour: Contour of the object.
    - number_radius (int): Number of radius values.
    - number_theta (int): Number of theta values.
    - max_radius (float): Maximum radius value.
    - center (tuple): Center coordinates of the grid.
    - desired_radius (float): Desired circle radius after transformation.

    Returns:
    - list: List of radius differences.
    """
    r = np.linspace(0, max_radius, number_radius)
    theta = np.linspace(0, 2 * np.pi, number_theta)

    # Compute distance between circle radius and object shape for each
    # discrete radius
    delta_r = []
    for i in theta:
        intersection = shapely.intersection(shapely.LineString(
            [center, [center[0] + max_radius * np.cos(i), center[1] + max_radius * np.sin(i)]]), object_contour)
        dr = shapely.distance(
            intersection,
            shapely.Point(center)) - shapely.distance(
            shapely.Point(
                [
                    center[0] + desired_radius * np.cos(i),
                    center[1] + desired_radius * np.sin(i)]),
            shapely.Point(center))
        delta_r.append(dr)
    return delta_r


def circular_warp(grids, image, order=0, verbose=False):
    """
    Perform circular warp transformation on the image using grids.

    Parameters:
    - grids (list): List containing original and transformed grids.
    - image: Input image to be transformed.
    - order: interpolation order see scipy.ndimage.map_coordinates.
    - verbose (bool): If True, display the transformed image.

    Returns:
    - numpy.ndarray: Transformed image.
    """
    src = np.dstack(grids[0])[0]
    dst = np.dstack(grids[1])[0]
    tform = FastPiecewiseAffineTransform()
    tform.estimate(dst, src)
    out = warp(image, tform, order=order)
    if verbose:
        plt.imshow(out)
    return out
