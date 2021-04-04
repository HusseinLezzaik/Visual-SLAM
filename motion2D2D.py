#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Motion estimation from 2 frames in monocular setup
 (Possibility to use the same algorithm to estimate the frame transfosrmation between left and right cameras in a stereo setup)
"""

import numpy as np
import cv2
import pptk # pip install pptk # !! Ubuntu users, pptk has a bug with libz.so.1, see how to do a new symlink in https://github.com/heremaps/pptk/issues/3

from draw import drawMatches


#%%
# Features detection and description
# Description vector is used to compare and match them between keyframes

gftt = cv2.GFTTDetector_create(maxCorners=1500) # alternate light detector
orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, WTA_K=4, scoreType=cv2.ORB_HARRIS_SCORE) # see https://docs.opencv.org/4.4.0/db/d95/classcv_1_1ORB.html#aeff0cbe668659b7ca14bb85ff1c4073b
det = orb # choice of feature detector algorithm
comp = orb # choice of feature descriptor algorithm
def detect(frame):
    kp = det.detect(frame) # find the keypoints
    return kp

def describe(frame, kp):
    kp, des = comp.compute(frame, kp) # compute descriptors
    return kp, des

#%%
# Features matching using their descriptions

# NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description ; with WTA_K=2, use NORM_HAMMING)
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=True) # see https://docs.opencv.org/4.4.0/d3/da1/classcv_1_1BFMatcher.html#ac6418c6f87e0e12a88979ea57980c020

def match(des1, des2):
    matches = bf.match(des1,des2)
    return matches


#%%
def dehomogenise(a_p4d):
    # changes an array of 4D points in homogeneous coordinates to 3D points in euclidean coordinates
    # input shape must be (N,4)
    # output shape is (N,3)
    return a_p4d[:,:3] / np.reshape(a_p4d[:,3], (-1,1))


#%%
def motion2d2d(cameraMat, frame1, frame2, kp1, kp2, des1, des2):
    # 2D-2D motion estimation from two images (from matching points in two images)
    
    # match keypoints between frames
    # TODO:
    matches1to2 = match(des1, des2)
    print("Number of matched points = {}".format(len(matches1to2)))
    
    # draw keypoints and matches
    #match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches1to2, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # original opencv implementation
    match_img, _ = drawMatches(frame1, kp1, frame2, kp2, matches1to2)
    cv2.imshow('matches', match_img)
    #return 0,0,0
    
    # place matched opencv keypoints' coordinaites in an array of [x,y] positions
    #a_kp1 = cv2.KeyPoint_convert(kp1) # built-in function to converts keypoints to list of coordinates
    #a_kp2 = cv2.KeyPoint_convert(kp2) # don't use it because only matched points are needed
    a_mkp1 = np.array([list(kp1[p.queryIdx].pt) for p in matches1to2])
    a_mkp2 = np.array([list(kp2[p.trainIdx].pt) for p in matches1to2])
    #print(a_mkp1.shape)

    # estimate Essential matrix, then decompose it in order to recover R and t
    #
    # TODO, use
    # cv2.findEssentialMat # see https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
    # cv2.recoverPose # see https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E, mask = cv2.findEssentialMat(a_mkp1, a_mkp2, cameraMat, method=cv2.RANSAC)
    print("Estimated E = {}".format(E))
    print("Number of inliers after RANSAC = {}".format(np.count_nonzero(mask))) # mask contains 0 for outliers and 1 for inliers
    #return 0,0,0
    
    # draw inlier matches
    inliers1to2 = [p for i, p in enumerate(matches1to2) if mask[i] == 1]
    #inliers_img = cv2.drawMatches(frame1, kp1, frame2, kp2, inliers1to2, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # original opencv implementation
    inliers_img, _ = drawMatches(frame1, kp1, frame2, kp2, inliers1to2, singlePointColor=[0.,0.,0.])
    #inliers_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches1to2, None, matchesMask=mask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # alternative with original opencv implementation
    cv2.imshow('inliers after RANSAC', inliers_img)
    #return 0,0,0
    
    nbinliers, R, t, mask2 = cv2.recoverPose(E, a_mkp1, a_mkp2, cameraMat, mask=mask.copy())
    print("Estimated R = {}".format(R))
    print("Estimated t = {}".format(t))
    print("Number of inliers after R,t estimation = {}".format(nbinliers)) # == np.count_nonzero(mask2)
    #return 0,0,0
    Rt = np.hstack((R,t)) # [R|t] matrix, 3x4 shape
    A = np.vstack((Rt, [0,0,0,1])) # transformation matrix in homogeneous coordinates
    
    # draw inlier matches
    inliers1to2 = [p for i, p in enumerate(matches1to2) if mask2[i] == 1]
    #inliers_img = cv2.drawMatches(frame1, kp1, frame2, kp2, inliers1to2, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # original opencv implementation
    inliers_img, rndcolors = drawMatches(frame1, kp1, frame2, kp2, inliers1to2, singlePointColor=[0.,0.,0.])
    #inliers_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches1to2, None, matchesMask=mask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # alternative with original opencv implementation
    cv2.imshow('inliers after R,t estimation', inliers_img)
    #return 0,0,0
    
    # lists of inliers kp
    a_ikp1 = a_mkp1[np.all(mask2 == 1, axis=1)]
    a_ikp2 = a_mkp2[np.all(mask2 == 1, axis=1)]
    #print(a_ikp1.shape)
    
    # now, 3D points of the map can be computed, at an arbitrary scale
    # world frame is the camera frame of the first picture
    # projection matrices of the camera at two positions P = K.[R|t], with K intrinsic parameters given in cameraMat
    #
    # TODO, use
    # cv2.triangulatePoints # see https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
    Rt_nothing = np.array([[1, 0, 0, 0], \
                           [0, 1, 0, 0], \
                           [0, 0, 1, 0]])
    projMatr1 = cameraMat @ Rt_nothing # 3x4 shape
    projMatr2 = cameraMat @ Rt # 3x4 shape
    p4d = ( cv2.triangulatePoints(projMatr1, projMatr2, a_ikp1.T, a_ikp2.T) ).T
    
    p3d = dehomogenise(p4d)
    #print(p3d.shape)
    
    return A, p3d, rndcolors


#%%
def main(frame1, frame2, cameraMat):
    
    # detect and describe keypoints
    # TODO:
    kp1 = detect(frame1)
    kp1, des1 = describe(frame1, kp1)
    kp2 = detect(frame2)
    kp2, des2 = describe(frame2, kp2)
    print("Number of keypoints in frame1 = {}".format(len(kp1)))
    print("Number of keypoints in frame2 = {}".format(len(kp2)))
    
    # show frame1
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    
    # show keypoints
    #kp_img = cv2.drawKeypoints(frame1, kp1, None)
    #cv2.imshow('keypoints', kp_img)
    #kp_img2 = cv2.drawKeypoints(frame2, kp2, None)
    #cv2.imshow('keypoints2', kp_img2)
    
    # estimate motion between 1 and 2
    M, p3d, rndcolors = motion2d2d(cameraMat, frame1, frame2, kp1, kp2, des1, des2)
    print("Motion homogeneous matrix (transformation matrix from 1 to 2) = {}".format(M))
    print(rndcolors.shape)
    # 4x4 matrix :
    # [R R R t]
    # [R R R t]
    # [R R R t]
    # [0 0 0 1]
    
    # show 3D points
    v = pptk.viewer(p3d)
    v.set(point_size=0.1)
    v.set(lookat=[0.,0.,0.])
    # apply the colors of matched points that are inliers (hence, the same that are triangulated to 3D)
    rndcolors = rndcolors[:,::-1] # be careful, opencv colors are orderd in BGR, so we need to put them in RGB for other libs
    v.attributes(rndcolors / 255.)


    while 1:
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            v.close()
            break


#%%
if __name__ == '__main__':
    # image pair for testing
    # test mono
    #im1 = cv2.imread("./kitti05/image_0/000660.png")
    #im2 = cv2.imread("./kitti05/image_0/000665.png")
    # test stereo
    im1 = cv2.imread("./kitti05/image_0/000660.png")
    im2 = cv2.imread("./kitti05/image_1/000660.png")
    #stereo_baseline = 0.54 # metres, setup described in http://www.cvlibs.net/datasets/kitti/setup.php
    print("Image shape = {}".format(im1.shape))
    
    # camera matrix recovered from Kitti calibration data
    # K = [[fx 0  cx]
    #      [0  fy cy]
    #      [0  0  1]]
    cameraMat = np.eye(3)
    cameraMat[0,0] = 7.070912000000e+02 # focal x
    cameraMat[1,1] = 7.070912000000e+02 # focal y
    cameraMat[0:2,2] = [6.018873000000e+02, 1.831104000000e+02] # principal point [cx, cy]
    #print(cameraMat)
    
    main(im1, im2, cameraMat)
