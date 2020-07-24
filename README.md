# Face-recognition-No-mistake-
#Face Recognition. Recognize and manipulate faces from Python or from the command line

#This is a project file for face recognition
import  cv2
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    return img, coords

def detect(img,faceCascade):
    img, coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 255), "Face")
    return img

capture = cv2.VideoCapture("Video.mp4")
while True:
    ret,frame = capture.read()
    frame =detect(frame, faceCascade)
    cv2.imshow("frame",frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
capture.release() #clear
cv2.destroyAllWindows()




