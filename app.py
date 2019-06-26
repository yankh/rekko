#!/usr/bin/env python
from flask import Flask, render_template, Response,request
import cv2
import keras
import numpy as np
import pickle
import dlib

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
print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh {}'.format(video))
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


# #Setting
# @app.route('/')
# def setting():
#     return render_template('login.html')





##end

# @app.route('/video',methods = ['POST'])
# def video():
#     #import
#     #from templates.nomdefichier import *




@app.route('/webcam')
def index():
    """Video streaming home page."""
    return render_template('webcam.html')

@app.route('/video/<string:path>')
def video(path):
    return render_template('video.html',path)

def gen(path=None):
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
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_local/<string:path>')
def video_feed_local(path):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(path=None),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
