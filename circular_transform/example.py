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

import circular_transform as CT
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
import sys
import os

# See https://medium.com/@bgallois/perfecting-imperfections-a-journey-in-warping-objects-into-circles-a0292e3c96ba
# Open image
original = cv2.cvtColor(
    cv2.imread(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/assets/orange.png"),
    cv2.COLOR_BGR2RGB)

# Find contour
_, image = cv2.threshold(original, 254, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(
    image[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea)
contours = np.squeeze(contours[-1])
contours[-1] = contours[0]  # close polygon
contours = shapely.LineString(contours)

# Parameters
params = dict()
params["number_radius"] = 15
params["number_theta"] = 50
params["max_radius"] = 600
params["center"] = (int(original.shape[1] // 2), int(original.shape[0] // 2))
params["desired_radius"] = 150

dr = CT.compute_radius_difference(
    object_contour=contours, **params)
grids = CT.get_transformation_grid(delta_r=dr, **params)
output = CT.circular_warp(grids, original, verbose=True)
cv2.imwrite(os.path.dirname(
    os.path.abspath(__file__)) +
    "/assets/orange_transformed.png", cv2.cvtColor(np.uint8(output * 255), cv2.COLOR_RGB2BGR))
plt.show()
