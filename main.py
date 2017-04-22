import numpy as np
import random
import cv2
from numpy import inf
import imutils

def corner_return(image):        
    image_cp = image.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
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
    return (x,y,w,h)

def findDoors(image):
    imagewithcorners,corn_loc= corner_return(image)
    cont1 =ret_contours(image,corn_loc)
    return cont1

def calc_desc(img,desc):
    x,y,w,h  = desc
    buffer = 0
    kp = fast.detect(img[y-buffer:y+h+buffer,x-buffer:x+w+buffer], None)
    return len(kp)


def mergeRect(rect1,rect2):
    bx,by,bw,bh = rect2
    x,y,w,h = rect1
    bx1=bx+bw
    by1=by+bh
    x1=x+w
    y1=y+h
    nx=0;ny=0;nx1=0;ny1=0;
    nx = x if x<bx else bx
    ny = y if y<by else by
    nx1 = x1 if x1>bx1  else bx1
    ny1 = y1 if y1>by1  else by1
    return (nx,ny,(nx1-nx),(ny1-ny))

def nonMaxSup(lBoxes,tresh):
    w = frame.shape[1]
    h = frame.shape[0]
    tmpImg = np.array(np.random.rand(h,w),np.int32)
    tmpImg.fill(0)
    
    for i in lBoxes.keys():
        x,y,w,h = lBoxes[i];
        tmpImg[y:y+h,x:x+w] = i
    fringe = lBoxes.keys()
    while len(fringe)>0:
        i = fringe.pop();
        if i in lBoxes.keys():
            x,y,w,h = lBoxes[i]
            x1=x+w
            y1=y+h
            cx = (x+x1)/2
            cy = (y+y1)/2
            noConf = True
            j=0;
            if (noConf and y1+tresh<h and set(tmpImg[y1:y1+tresh,cx].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,cx].flatten())
            if (noConf and y-tresh>0 and set(tmpImg[y-tresh:y,cx].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,cx].flatten())
            if (noConf and x-tresh>0 and set(tmpImg[cy,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[cy,x-tresh:x].flatten())
            if (noConf and x1+tresh<w and set(tmpImg[cy,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[cy,x1:x1+tresh].flatten())
            if (noConf and y1+tresh<h and x1+tresh<w and set(tmpImg[y1:y1+tresh,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,x1:x1+tresh].flatten())
            if (noConf and y-tresh>0 and x-tresh>0 and set(tmpImg[y-tresh:y,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,x-tresh:x].flatten())
            if (noConf and y1+tresh<h and x-tresh>0 and set(tmpImg[y1:y1+tresh,x-tresh:x].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y1:y1+tresh,x-tresh:x].flatten())
            if (noConf and y-tresh>0 and x1+tresh<w and set(tmpImg[y-tresh:y,x1:x1+tresh].flatten())!=set([0])):
                noConf = False
                confZone=set(tmpImg[y-tresh:y,x1:x1+tresh].flatten())
            if not noConf:
                confZone.discard(0)
                confZone.discard(i)
                for j in confZone:
                    lBoxes[i] = mergeRect(lBoxes[i],lBoxes[j])
                    (nx,ny,nw,nh) = lBoxes[i]
                    nx1 = nx+nw
                    ny1 = ny+nh
                    tmpImg[ny:ny1,nx:nx1] = i
                    #print x,y,x1,y1,bx,by,bx1,by1,nx,nx1,ny,ny1
                    #print i,j,lBoxes.keys()
                    lBoxes.pop(j)
                    contList[i].extend(contList[j])
                    if j in fringe:
                        fringe.remove(j)
                fringe.append(i)
                #print "dsadas",i,j,lBoxes.keys()
    return lBoxes


def isValidBox(box):
    x,y,w,h = box
    if ((h>70 and h<150) or (w>70 and w<150)):
        if calc_desc(frame,(x,y,w,h))>80:
            return True;
    return False;

def find_contor(label,k):
    kernel = np.ones((10,10),np.uint8)
    label_tmp = label == k
    label_tmp = label_tmp.astype(np.uint8)
    label_tmp = label_tmp.reshape(frameRight.shape)
    ret, thresh = cv2.threshold(label_tmp, 127, 255, 0)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_CLOSE, kernel)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(label_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getposition(box):
    pos=""
    x,y,w,h  = box
    xc=x+w/2
    yc=y+h/2
    if yc-240<-80:
        pos+="T"
    if yc-240>80:
        pos+="B"
    if xc-320<-107:
        pos+="L"
    if xc-320>107:
        pos+="R"
    if pos=="":
        pos="C"
    return pos

def getDistancewithContour(cnts,box,disp):
    points = []
    x,y,w,h  = box
    dist = []
    infCount=0
    numOfPoints = 5
    while(len(points)<numOfPoints):
        xp = random.randint(x,x+w)
        yp = random.randint(y,y+h)
        #print(xp,yp)
        #print(cv2.pointPolygonTest(cnt,(xp,yp),False))
        #print len(cnts)
        #print [cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts]
        isValidPoint = any([cv2.pointPolygonTest(cnt,(xp,yp),False)>=0 for cnt in cnts])
        if isValidPoint and (yp,xp) not in points :
            points.append((yp,xp))
            if disp[yp,xp]==0:
                infCount+=1
            else:
                dist.append(bf/disp[yp,xp])
    final = [x for x in dist if abs(x-np.mean(dist)) <= 1 * np.std(dist)]
    if infCount < numOfPoints-numOfPoints/2 and len(final)>0:
        retDist = sum(final)/len(final)
        return retDist
    else:
        return inf

cap = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)
leftInt = np.array([[916.5253,0,386.6512],[0,952.5333,194.1781],[0, 0,1.0000]])
rightInt = np.array([[997.3000,0,231.4237],[0,980.8277,126.2737],[0,0,1.0000]])
rightExt = np.array([0.0172,-0.4886,-0.0080,-0.0053,2.5232])
leftExt = np.array([0.1186,-5.6059,0.0165,0.0364,22.0870])

f = (leftInt[0][0] + leftInt[1][1] +  rightInt[0][0] + rightInt[1][1])/40.0
b = 30

bf = f*b

cap.set(cv2.cv.CV_CAP_PROP_FPS,3);
cap1.set(cv2.cv.CV_CAP_PROP_FPS,3);

ret,frameRight = cap.read()
a = np.array([range(0,frameRight.shape[1]),range(0,frameRight.shape[1])])
col = np.repeat(a,frameRight.shape[0]/2,axis=0)
b = np.array([range(0,frameRight.shape[0]),range(0,frameRight.shape[0])])
row = np.repeat(b,frameRight.shape[1]/2,axis=0)
row = row.transpose()
row = row.reshape((-1,1))
col = col.reshape((-1,1))
row = np.float32(row)
col = np.float32(col)
fast = cv2.FastFeatureDetector()
orb = cv2.ORB()

while(True):
    ret,frameRight = cap.read()
    frame = frameRight 
    frameRight = cv2.medianBlur(frameRight,9)
    frameRight = cv2.cvtColor(frameRight,cv2.COLOR_BGR2GRAY)
    h,  w = frameRight.shape[:2]
    #newcameramtxRight, roi=cv2.getOptimalNewCameraMatrix(frameRight,rightExt,(w,h),1)
    frameRight = cv2.undistort(frameRight, rightInt, rightExt, None,None)

    
    ret,frameLeft = cap1.read()
    frameLeft = cv2.medianBlur(frameLeft,9)
    frameLeft=cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
    h,  w = frameLeft.shape[:2]
    #newcameramtxLeft, roi=cv2.getOptimalNewCameraMatrix(frame1,leftExt,(w,h),1,(w,h))
    frameLeft = cv2.undistort(frameLeft, leftInt, leftExt, None, None)


    #stereo = cv2.StereoSGBM(0, 96, 5, 600, 2400, 20, 16, 1,  100, 20,True)
    stereo = cv2.StereoSGBM(0, 64, 10, 600, 2400, 20, 16, 1,  100, 20,True)
    #stereo = cv2.StereoBM(1, 16, 15)
    disparityMat = stereo.compute(frameLeft,frameRight,cv2.CV_32F)
    disparity = cv2.convertScaleAbs(disparityMat)
    
    
    kp = fast.detect(frameRight, None)
    #kp = orb.detect(frame,None)
    #kp, des = orb.compute(frame, kp)
    Z = frame.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    #Kvalues = np.column_stack((row,col,Z))
    Kvalues = Z
    Kvalues = np.array(Kvalues)
    #keypoints = detector.detect(frameRight)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Kvalues,K,criteria,10,cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    #center = center[:,[2,3,4]]
    contors = []
    for i in range(0,K):
        cont = find_contor(label,i)
        contors.append(cont)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    test_cont = find_contor(label,2)

    boxes = {}
    distList = {}
    index =0;
    contList = {}
    for i in range(0,K):
        tmp_cont = np.asarray(contors[i])
        for j in range(0,tmp_cont.shape[0]):
            cnt = tmp_cont[j]
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
            x,y,w,h = cv2.boundingRect(cnt)
            box_temp = (x,y,w,h)
            if isValidBox(box_temp):
                dist = getDistancewithContour(cnt,box_temp,disparity)
                if dist != inf:
                    distList[index] = dist
                    boxes[index] = box_temp
                    contList[index] = [cnt]
                    index+=1;

    #for i in boxes.keys():
    #    if not isValidBox(boxes[i]):
    #        boxes.pop(i)
    boxes = nonMaxSup(boxes,20)
    for i in boxes.keys():
        x,y,w,h = boxes[i]
        pos = getposition(boxes[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(disparity,(x,y),(x+w,y+h),(0,255,0),2)
        x,y,w,h = findDoors(frame)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        dist = getDistancewithContour(contList[i],boxes[i],disparity)
        if(dist!=inf):
            cv2.putText(frame,str(int(dist)),(x+w/2,y+h/2),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255))
        #else:
        #    if calc_desc(frame,(x,y,w,h))>10:
        #        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #epsilon = 0.1*cv2.arcLength(cnt,True)
        #approx = cv2.approxPolyDP(cnt,epsilon,True)
        #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
    #cv2.drawContours(frame, test_cont, -1, (0,255,0), 1)
    
    #frame = cv2.drawKeypoints(frame, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame',frame)
    cv2.imshow('disparity',disparity)
    '''
    # find and draw the keypoints
    kp = fast.detect(gray, None)
    keypoints = detector.detect(gray)
    print(keypoints)
    im = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame',im)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
