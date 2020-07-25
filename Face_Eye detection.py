import  cv2
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyewithglassCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
# mouthCascade = cv2.CascadeClassifier("Mouth.xml")


def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        coords= [x,y,w,h]
    return img, coords

def detect(img,faceCascade,eyewithglassCascade):
    img,coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 255), "Face")
    img,coords = draw_boundary(img, eyewithglassCascade, 1.1, 15, (155, 20, 55), "Eye")
    # img, coords = draw_boundary(img, mouthCascade, 1.1, 30, (15, 20, 55), "Mouth")
    return img

capture = cv2.VideoCapture("BNK.mp4")
while True:
    ret,frame = capture.read()
    frame =detect(frame, faceCascade,eyewithglassCascade)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release() #clear
cv2.destroyAllWindows()


