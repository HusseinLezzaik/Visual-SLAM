#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified version of opencv DrawMatches to recover randomly generated colors
"""

import numpy as np
import cv2

draw_shift_bits = 4
draw_multiplier = 1 << draw_shift_bits

def _drawKeypoint( img, p, color ):
    assert len(img) > 0
    center = (round(p.pt[0]) * draw_multiplier, round(p.pt[1]) * draw_multiplier)
    
    #radius = round(p.size/2 * draw_multiplier)
    radius = 3 * draw_multiplier
    img = cv2.circle( img, center, radius, color, thickness=1, lineType=cv2.LINE_AA, shift=draw_shift_bits )
    
    return img

def drawKeypoints( image, keypoints, color = None ):
    image = np.reshape(image, (image.shape[0], image.shape[1], -1))
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR())

    isRandColor = color == None

    outImage = image.copy()
    for k in keypoints:
        if isRandColor:
            color = np.random.randint(0,256, size=(3)).astype(np.double) # int values but stored as doubles to fit opencv Scalar objects
        outImage = _drawKeypoint( outImage, k, color )

    return outImage

def _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2, singlePointColor = None):
    img1 = np.reshape(img1, (img1.shape[0], img1.shape[1], -1))
    img2 = np.reshape(img2, (img2.shape[0], img2.shape[1], -1))
    if img1.shape[2] == 1:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR())
    if img2.shape[2] == 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR())
    img1size = img1.shape
    img2size = img2.shape
    #size = [ img1size[0] + img2size[0], MAX(img1size[1], img2size[1]) ] # vertical
    size = [ max(img1size[0], img2size[0]), img1size[1] + img2size[1] ] # horizontal
   
    out_cn = 3
    outImg = np.zeros((size[0], size[1], out_cn), dtype=np.uint8)

    outImg1 = img1.copy()
    outImg2 = img2.copy()
    
    outImg1 = drawKeypoints( outImg1, keypoints1, singlePointColor )
    outImg2 = drawKeypoints( outImg2, keypoints2, singlePointColor )
    
    outImg[0:img1size[0],0:img1size[1]] = outImg1
    outImg[0:img2size[0],img1size[1]:img1size[1]+img2size[1]] = outImg2
    
    return outImg, outImg1, outImg2

def _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor = None ):
    # added a way to output the color for later use on 3d points
    
    isRandMatchColor = matchColor == None
    if isRandMatchColor:
        color = np.random.randint(0,256, size=(3)).astype(np.double) # int values but stored as doubles to fit opencv Scalar objects
    else:
        color = matchColor

    outImg1 = _drawKeypoint( outImg1, kp1, color )
    outImg2 = _drawKeypoint( outImg2, kp2, color )
    img1size = outImg1.shape
    img2size = outImg2.shape
    outImg[0:img1size[0],0:img1size[1]] = outImg1
    outImg[0:img2size[0],img1size[1]:img1size[1]+img2size[1]] = outImg2

    pt1 = kp1.pt
    pt2 = kp2.pt
    dpt2 = (min(pt2[0]+outImg1.shape[1], float(outImg.shape[1]-1)), pt2[1])

    outImg = cv2.line( outImg, \
                      (round(pt1[0]*draw_multiplier), round(pt1[1]*draw_multiplier)), \
                      (round(dpt2[0]*draw_multiplier), round(dpt2[1]*draw_multiplier)), \
                      color, 1, cv2.LINE_AA, draw_shift_bits )
    outImg1 = outImg[0:img1size[0],0:img1size[1]]
    outImg2 = outImg[0:img2size[0],img1size[1]:img1size[1]+img2size[1]]
    
    return outImg, outImg1, outImg2, color

def drawMatches( img1, keypoints1, img2, keypoints2, matches1to2, matchColor = None, singlePointColor = None, matchesMask = None ):
    # added a way to output the random  colors for later use on 3d points
    
    assert matchesMask == None or matchesMask.shape[0] == matches1to2.shape[0]

    outImg, outImg1, outImg2 = _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2, singlePointColor )

    # draw matches
    rndcolors = np.array([])
    for (i,m) in enumerate(matches1to2):
        i1 = m.queryIdx;
        i2 = m.trainIdx;
        if matchesMask == None or matchesMask[i] == 1:
            kp1 = keypoints1[i1]
            kp2 = keypoints2[i2]
            outImg, outImg1, outImg2, color = _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor )
            if len(rndcolors) == 0:
                rndcolors = color.copy()
            else:
                rndcolors = np.vstack((rndcolors, color))

    return outImg, rndcolors