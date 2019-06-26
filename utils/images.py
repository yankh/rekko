import cv2
import keras
import numpy as np
import pickle
import dlib

class Images():

    def __init__(self,path):
        self.path = path
        self.img = cv2.imread(path)

    def predictframe(self,frame):
        try:

            frame=cv2.resize(frame,(64,64))
            test=np.array(np.float16(frame/255))
            test=np.expand_dims(test,axis=0)
            hist=model_final.predict(test)
            pred=svclassifier.predict(hist)
            prob=svclassifier.predict_proba(hist)
            p=max(prob[0])
            return pred,p
        except:
            return 'None',0

    def rect_to_bb(self,rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)  

    def img_face_detect(self):
        faceslist=[]
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
            x, y, w, h=rect_to_bb(rect)
            visage=self.img[y:y+h,x:x+w]
            text,p=predictframe(visage)
            cord=[x,y,w,h]
            ab=[visage,cord,text[0],p]
            
            faceslist.append(ab)
        
        return faceslist