#!/usr/bin/env python
from flask import Flask, render_template, Response,request
import cv2
import keras
import numpy as np
import pickle
import dlib
from utils.images import Images

model_final=keras.models.load_model('D:/MIV/PFE/Application Interface/m.h5')
model_final._make_predict_function()
#model_final.load_weights('desc_weights.h5')

svclassifier=pickle.load(open('D:/MIV/PFE/Application Interface/svm_desc_baseP.pickle','rb'))

def predictframe(frame):
    try:
        frame=cv2.resize(frame,(224,224))
        test=np.array(np.float16(frame/255))
        test=np.expand_dims(test,axis=0)
        hist=model_final.predict(test)
        pred=svclassifier.predict(hist)
        prob=svclassifier.predict_proba(hist)
        p=max(prob[0])
        return pred,p
    except:
        return 'None',0
    

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)    

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
video = cv2.VideoCapture(0)
##yanis front impl
#Logout page displaying
@app.route('/')
def logout():
    return render_template('login.html')

# Login page displaying
@app.route('/', methods = ['POST'])
def login():
    if request.method == 'POST' and 'login' in request.form:
        if (request.form['password'] == "popo") and (request.form['username'] == "admin"):
            return render_template('admin.html')
        else:
            return render_template('login.html', wrong=True)



@app.route('/webcam')
def index():
    """Video streaming home page."""
    return render_template('webcam.html')

def gen(local=False):
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
            x, y, w, h=rect_to_bb(rect)
            visage=frame[y:y+h,x:x+w]
            text,p=predictframe(visage)
            #print(text,'/',p)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame,text[0]+'/'+str(p), 
                        (x,y-5), 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
                    
def gen_local():
    cap = cv2.VideoCapture('C:/m.mp4')
    print(cap)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame, -1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
            
                x, y, w, h=rect_to_bb(rect)
        
                visage=frame[y:y+h,x:x+w]
                text,p=predictframe(visage)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,text[0]+'/'+str(p), 
                            (x,y-5), 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
            cv2.imwrite('p.jpg', frame)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + open('p.jpg', 'rb').read() + b'\r\n')

def gen_local_image(path='.'):
    img = Images(path)
    frame = img.predict()
    cv2.imwrite('t.jpg', frame)
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_local')
def video_local():
    return Response(gen_local(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_local/<string:path>')
def image_local(path):
    return Response(gen_local_image(path),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)

