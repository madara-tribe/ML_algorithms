import math
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.feature import hog
import cv2
from sklearn.svm import SVC
import joblib
import json

from HogSVM import crop_query, load_dataset, train_svm_model, test_svm_model


def homography_transformation(query, original, by_rotation=True):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    im1Gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(query, keypoints1, original, keypoints2, matches, None)
    plt.imshow(imMatches,),plt.show()

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    angle = math.atan2(h[0,1], h[0,0]) * 180 / math.pi
    if by_rotation:
        height, width=query.shape[:2]
        rotation_matrix=cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        modifid_img=cv2.warpAffine(query, rotation_matrix,(width,height))
        print('height and width are', width)
        print('convert_angle', angle)
    else:
        height, width, channels = original.shape
        modifid_img = cv2.warpPerspective(query, h, (width, height))

    plt.imshow(modifid_img,),plt.show()
    return modifid_img



def modify_position_by_angle():
    output_dir = 'database'
    print('search query sim image from DB')
    # load query
    filename = 'database/query1.jpeg'
    query = crop_query(filename)
    hs, ws = query.shape[:2]

    # make dataset
    trains=load_dataset('database', hs)
    print(query.shape, trains.shape)

    # train svm
    sim_query, detector = train_svm_model(query, trains)

    print('test svm model')
    tests = rotation(trains)
    test_svm_model(query, tests, detector=detector)


    print('angle potisition corecction of sim query image')
    original = 'database/original.jpg'
    original_img = cv2.imread(original)
    print(original_img.shape)
    sim_query = cv2.resize(sim_query, (original_img.shape[0], original_img.shape[1]))
    print(sim_query.shape)

    modifid_img = homography_transformation(sim_query, original_img, by_rotation=True)
    modifid_ = modifid_img.astype(np.uint8)
    rgb_modifid = cv2.cvtColor(modifid_, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, "modifid_query.png"), rgb_modifid)

if __name__ == '__main__':
    modify_position_by_angle()
