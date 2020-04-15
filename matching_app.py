

import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img,cmap='gray'):
    
    fig = plt.figure(figsize=(25,17))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.show()


for i in range(1,8):
    
    reeses = cv2.imread(f'book{i}.jpg')
    cereals = cv2.imread('booklist1.png') 

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(reeses,None)
    kp2, des2 = sift.detectAndCompute(cereals,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=5)  

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test
    for i,(match1,match2) in enumerate(matches):
        if match1.distance < 0.45*match2.distance:
            matchesMask[i]=[1,0]
    y=0
    for a,b in matchesMask:
        if a == 1:
            y +=1

    print(y)

    draw_params = dict(matchColor = (150,0,0,),
                       matchesMask = matchesMask,
                       flags = 2)

    flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)

    display(flann_matches)