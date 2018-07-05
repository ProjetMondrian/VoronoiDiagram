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
import random
from datetime import datetime



# Import image
load_path = os.path.join(os.getcwd(), 'Images', 'women.jpg');
save_path = os.path.join(os.getcwd(), 'Images', 'exampleresult5.jpg');
img = io.imread(load_path)

# Retrieve facial landmarks
points = facial_landmarks(img)
minx = np.amin(points[:,0])
miny = np.amin(points[:,1])
maxx = np.amax(points[:,0])
maxy = np.amax(points[:,1])
img = img[minx:maxx,miny:maxy]

# rescale points coordinates
n = points.shape[0]
for i in range(0,n):
    points[i,0] = points[i,0] - minx
    points[i,1] = points[i,1] - miny

# Compute Voronoi partition
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# Colorize the picture
for region in regions:

    # find points inside each region
    poly = vertices[region]
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)

    # find dominant color using k means clustering
    colors = img[rr, cc]
    if colors.shape[0]>0:
        kmeans = KMeans(n_clusters=1, random_state=0).fit(colors)
        c = kmeans.cluster_centers_
        r = int(round(c[0,0]))
        g = int(round(c[0,1]))
        b = int(round(c[0,2]))

        # colorize the inside
        img[rr, cc] = (r, g, b)

        # colorize the perimeter
        rr, cc = polygon_perimeter(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc] = (0, 0, 0)



# Crop and save
io.imsave(save_path,img)
