import numpy as np
from scipy.spatial import Voronoi
from finite_voronoi import voronoi_finite_polygons_2d
from landmarks import land2coords, facial_landmarks
import os
import sys
from skimage import io, color
from skimage.draw import polygon, polygon_perimeter
from random import randint
from sklearn.cluster import KMeans
import dlib
import random
from datetime import datetime
from skimage.exposure import equalize_adapthist
from collections import Counter
from skimage.color import rgb2lab,lab2rgb
from skimage.segmentation import mark_boundaries, slic




# Import image
load_path = sys.argv[1]
facial_detection = sys.argv[2]
if len(sys.argv) > 3:
    npoints = sys.argv[3]

save_path = "FinalResultDavid.jpg"
img = io.imread(load_path)

if not facial_detection :
    x = np.random.randint(img.shape[0], size=npoints)
    y = np.random.randint(img.shape[1], size=npoints)
    points = np.zeros((npoints,2), dtype=int)
    points[:,0] = x
    points[:,1] = y

else :
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
vor = Voronoi(points,incremental = 1)
vertices = vor.vertices
regions = vor.regions


# compute finite voronoi
regions, vertices = voronoi_finite_polygons_2d(vor)

# Colorize the picture
img = rgb2lab(img) # to lab colorspace

for region in regions:

    # find points inside each region
    poly = vertices[region]
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)

    # find dominant color using k means clustering
    colors = img[rr, cc]
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(colors)
    #count labels to find most popular
    label_counts = Counter(labels)
    #subset out most popular centroid
    dominant_color = kmeans.cluster_centers_[label_counts.most_common(1)[0][0]]
    x = dominant_color[0]
    y = dominant_color[1]
    z = dominant_color[2]

    # colorize the inside
    img[rr, cc] = (x, y, z)

    # colorize the perimeter
    rr, cc = polygon_perimeter(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc] = (0, 0, 0)

# Back to rgb and save
img = lab2rgb(img)
io.imsave(save_path,img)
