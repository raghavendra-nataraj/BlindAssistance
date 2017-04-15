import numpy as np
import cv2

def find_contor(label,k):
    kernel = np.ones((10,10),np.uint8)
    label_tmp = label == k
    label_tmp = label_tmp.astype(np.uint8)
    label_tmp = label_tmp.reshape(gray.shape)
    ret, thresh = cv2.threshold(label_tmp, 127, 255, 0)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_CLOSE, kernel)
    res2 = cv2.morphologyEx(label_tmp, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(label_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calc_desc(img,desc):
    x,y,w,h  = desc
    kp = fast.detect(img[y-10:y+h+10,x-10:x+w+10], None)
    return len(kp)


cap = cv2.VideoCapture(1)
ret,frame = cap.read()
a = np.array([range(0,frame.shape[1]),range(0,frame.shape[1])])
col = np.repeat(a,frame.shape[0]/2,axis=0)
b = np.array([range(0,frame.shape[0]),range(0,frame.shape[0])])
row = np.repeat(b,frame.shape[1]/2,axis=0)
row = row.transpose()
row = row.reshape((-1,1))
col = col.reshape((-1,1))
row = np.float32(row)
col = np.float32(col)
fast = cv2.FastFeatureDetector()
orb = cv2.ORB()

while(True):
    ret,frame = cap.read()
    #frame = cv2.cvtColor(frames,cv2.COLOR_BGR2LAB)
    frame = cv2.medianBlur(frame,9)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)    
    kp = fast.detect(gray, None)
    #kp = orb.detect(frame,None)
    #kp, des = orb.compute(frame, kp)
    Z = frame.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    #Kvalues = np.column_stack((row,col,Z))
    Kvalues = Z
    Kvalues = np.array(Kvalues)
    #keypoints = detector.detect(gray)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 6, 1.0)
    K = 5
    ret,label,center=cv2.kmeans(Kvalues,K,criteria,10,cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    #center = center[:,[2,3,4]]
    contors = []
    print("dasda",len(contors))
    for i in range(0,K):
        cont = find_contor(label,i)
        print("dasda",len(cont))
        contors.append(cont)
    print(len(contors))
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    test_cont = find_contor(label,2)

    for i in range(0,K):
        tmp_cont = np.asarray(contors[i])
        for j in range(0,tmp_cont.shape[0]):
            cnt = tmp_cont[j]
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
            x,y,w,h = cv2.boundingRect(cnt)
            if ((h>70 and h<150) or (w>70 and w<150)):
                if calc_desc(frame,(x,y,w,h))>10:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #else:
            #    if calc_desc(frame,(x,y,w,h))>10:
            #        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #epsilon = 0.1*cv2.arcLength(cnt,True)
            #approx = cv2.approxPolyDP(cnt,epsilon,True)
            #cv2.drawContours(frame, [cnt], -1, (0,255,0), 1)
    #cv2.drawContours(frame, test_cont, -1, (0,255,0), 1)
    
    frame = cv2.drawKeypoints(frame, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('res2',frame)
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
