#!pip install opencv-python==3.4.2.16 && pip install opencv-contrib-python==3.4.2.16
# sift = cv2.xfeatures2d.SIFT_create()
import sys
import numpy as np
import cv2
import os
from keras.preprocessing import image

# python modefy_image_position.py original-image curved-image
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def modify_position(curved, original):

  im1Gray = cv2.cvtColor(curved, cv2.COLOR_BGR2GRAY)
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
  imMatches = cv2.drawMatches(curved, keypoints1, original, keypoints2, matches, None)
  # plt.imshow(imMatches,),plt.show()
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, channels = original.shape
  modifid_img = cv2.warpPerspective(curved, h, (width, height))
  # plt.imshow(modifid_img,),plt.show()
  return modifid_img


if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)

    original_name = argvs[1]
    curved_name = argvs[2]

    # load and resie
    original = cv2.imread(original_name)
    curved = cv2.imread(curved_name)
    curved = cv2.resize(curved, (original.shape[0],original.shape[1]))
    # modefy position
    imreg = modify_position(curved, original)
    save_dir = '/Users/downloads'
    print("Saving aligned image...")
    cv2.imwrite(os.path.join(save_dir,'modeied.png'), imreg)
    #img = image.array_to_img(imreg, scale=False)
    #img.save(os.path.join(save_dir, 'modified' + '.png'))
