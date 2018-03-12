"""
Created on Wed Jan  3 16:05:20 2018

@author: Joseph
"""
#import tensorflow as tf
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import argparse
#import xfeatures2d

#inlier_threshold=2.5;#Distance threshold to identify inliers
#nearest_neighbor_matchratio=0.7;#Nearest neighbor matching ratio
# import the necessary packages
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
draw_point=0
draw_x=-1
draw_y=-1

refPt = []
def detect_click(event,x,y,flags,param):
    global refPt,draw_x,draw_y,draw_point
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt.append((x,y))
        print(x)
        print(y)
    elif event ==cv2.EVENT_LBUTTONUP:
        print("reference points:")
        print(refPt)
        draw_point=1
        draw_x=x
        draw_y=y
        
        #cv2.PutText(img, ".", (x,y), Helvetica, Black)
        


#Initialize reference points
def main():
    global refPt,draw_x,draw_y,draw_point
    # find the keypoints and descriptors with ORB
    #1.1) Instantiate video
    video = cv2.VideoCapture(0)
    ret, img = video.read()
    cv2.namedWindow("image")
    print(cv2.__version__)
    
    cv2.setMouseCallback("image", detect_click)
    
    
    while(True):
        cv2.imshow('image',img)
        key = cv2.waitKey(2) & 0xFF
        if draw_point==1:
            cv2.circle(img,(draw_x,draw_y),5,(0,0,255),-1)
            draw_point=0
            
        elif key==ord('q'):
            break
        elif len(refPt)>=2:
            break
    
    img_orig=img

    
    #Find new cropped image
    print(refPt)
    left_x=refPt[0][0] if refPt[0][0]<refPt[1][0] else refPt[1][0]
    top_y=refPt[0][1] if refPt[0][1]<refPt[1][1] else refPt[1][1]
    right_x=refPt[0][0] if refPt[0][0]>refPt[1][0] else refPt[1][0]
    bot_y=refPt[0][1] if refPt[0][1]>refPt[1][1] else refPt[1][1]

    crop_img = img_orig[top_y:bot_y, left_x:right_x]
    
    
    orb = cv2.ORB_create()
    kp_crop, des_crop = orb.detectAndCompute(crop_img,None)
    while(True):
        
        ret, img2 = video.read()
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#Use norm distance since binary string-based discriptors use norm hamming (ORB, BRIEF, BRISK)
        # Initiate SIFT detector
        # Match descriptors.
        matches = brute_force_matcher.match(des_crop,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3=img_orig
        img3 = cv2.drawMatches(crop_img,kp_crop,img2,kp2,matches[:5], img3,flags=2)
        color = cv2.cvtColor(img3, cv2.cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0
    #Up to here
main()
