import dlib
import numpy as np
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.transform import resize

# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (a, b)-coordinates
    return coords


def facial_landmarks(img):
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # convert to grayscale for better efficiency and reshape
    n = img.shape[0]
    m = img.shape[1]
    if n>200 and m>200:
        img_gray = resize(img, (400,400))
    img_gray = rgb2gray(img_gray)
    img_gray = img_as_ubyte(img_gray)

    # detecting faces
    points = []
    face_boundaries = face_detector(img_gray,0)
    for (enum,face) in enumerate(face_boundaries):
        # Now when we have our ROI(face area) let's
        # predict and draw landmarks
        landmarks = landmark_predictor(img_gray, face)
        # converting co-ordinates to NumPy array
        landmarks = land2coords(landmarks)
        points.append(landmarks)

    if len(points) > 0:
        points = np.array(points)
        points = points[0,:,:]
        points = points.astype(int)
        l = points.shape[0]

        for i in range(0,l):
            tmp = points[i,0]
            points[i,0] = points[i,1]
            points[i,1] = tmp

        # rescale points
        if n>200 and m>200:
            for i in range(0,l):
                points[i,0] = (n/400)*points[i,0]
                points[i,1] = (m/400)*points[i,1]

        return points
    else:
        points = np.zeros((1,2))
        return points
