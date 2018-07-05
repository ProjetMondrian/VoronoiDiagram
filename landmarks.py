import dlib
import numpy as np

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

    # detecting faces
    points = []
    face_boundaries = face_detector(img,0)
    for (enum,face) in enumerate(face_boundaries):
        # Now when we have our ROI(face area) let's
        # predict and draw landmarks
        landmarks = landmark_predictor(img, face)
        # converting co-ordinates to NumPy array
        landmarks = land2coords(landmarks)
        points.append(landmarks)

    points = np.array(points)
    points = points[0,:,:]
    points = points.astype(int)
    n = points.shape[0]

    for i in range(0,n):
        tmp = points[i,0]
        points[i,0] = points[i,1]
        points[i,1] = tmp

    return points
