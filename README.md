Algorithm : extract facial landmarks from the picture, use them as seed points (along with some randomly chosen points) for a voronoi partition
Requirements : scipy : pip install scipy
skimage : pip install skimage
sklearn : pip install sklearn
dlib : pip install dlib (you also need cmake to get the installation working)

Challenges : the partition produces regions that are not square
not useful when the image does not contain a face
improve speed

