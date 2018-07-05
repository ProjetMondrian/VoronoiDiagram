import numpy as np
from scipy.spatial import Voronoi
from finite_voronoi import voronoi_finite_polygons_2d
from landmarks import land2coords, facial_landmarks
import os
from skimage import io, color
from skimage.draw import polygon, polygon_perimeter
from random import randint
from sklearn.cluster import KMeans
import dlib
from skimage.exposure import adjust_gamma
import random



# Import image
load_path = os.path.join(os.getcwd(), 'Images', 'obama.jpg');
save_path = os.path.join(os.getcwd(), 'Images', 'result.jpg');
img = io.imread(load_path)

# Retrieve facial landmarks
points = facial_landmarks(img)

# Add some other random points
npoints = 5
x = np.random.randint(img.shape[0], size=npoints)
y = np.random.randint(img.shape[1], size=npoints)
pointsrnd = np.zeros((npoints,2), dtype=int)
pointsrnd[:,0] = x
pointsrnd[:,1] = y
points = np.concatenate((points,pointsrnd))

# Compute Voronoi partition
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# Colorize the picture
for region in regions:

    # find points inside each region
    poly = vertices[region]
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)

    # find dominant color using k means clustering
    colors = img[rr, cc, :]
    if colors.shape[0]>0:
        kmeans = KMeans(n_clusters=1, random_state=0).fit(colors)
        c = kmeans.cluster_centers_
        r = int(round(c[0,0]))
        g = int(round(c[0,1]))
        b = int(round(c[0,2]))

        # colorize the inside
        img[rr, cc, :] = (r, g, b)

        # colorize the perimeter
        rr, cc = polygon_perimeter(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc, :] = (0, 0, 0)


# Save picture
io.imsave(save_path,img)
