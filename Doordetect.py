# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:27:05 2017

@author: Arun Ram
"""

import os
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
import scipy
from scipy import spatial
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils



os.chdir('C:\\Users\\Arun Ram\\Desktop\\Vision project')
        
def corner_return(image):
        
    image_cp = image.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    shp = max(image.shape)
    
    
   
    
    scl = 700 /shp
    image =cv2.resize(image,None, fx=scl, fy=scl)
    image =np.float32(image)
    
    
    blr = cv2.GaussianBlur(image,(5,5),0)
    #blr =cv2.bilateralFilter(image,9,40,75)
    
    rs= cv2.goodFeaturesToTrack(blr,20,0.03,20,None,None,2,useHarrisDetector=True,k=0.04)
    
                                
    #r,c,d= rs.shape
    
    #rs=np.int0(rs)
    
    for k in rs:
        x,y= k.ravel()
        cv2.circle(image,(x,y),5,127,-1)
    

    
   
    #cv2.circle(image,(229,400),20,250,-1)     
                                
    return image,rs






    
    
    
    
def ret_contours(image,locs):
    
    image_cp = image.copy()
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        
    blr =cv2.bilateralFilter(image,9,75,75)
    edges= cv2.Canny(blr,40,65,apertureSize = 3)
    
    edges = cv2.dilate(edges,None, iterations=1)
    edges = cv2.erode(edges,None, iterations=1)
    
    
    cont = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    cont = cont[0] if imutils.is_cv2() else cont[1]
    #(cont,_) = contours.sort_contours(cont)
    max_val=0
    k=0
   
    for a in cont:
        m=0
        x,y,w,h = cv2.boundingRect(a)
    
        if ((h>=(2*w)) and (h>400) and (w>10)):
            m=h
            #image_cp1 = image_cp.copy()
            cv2.rectangle(image_cp,(x,y),(x+w,y+h),(0,255,0),2)

    return image_cp
               
               
   









image  = cv2.imread('4.jpg')

#image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)






imagewithcorners,corn_loc= corner_return(image)


cont1 =ret_contours(image,corn_loc)





cv2.imwrite('resutgot.jpg',cont1)


