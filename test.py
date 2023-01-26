import cv2

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(0)
ret = True
while ret:
    ret,frame = cap.read()
    #if extract_faces(frame)!=():
    if ret != False:
        cv2.putText(frame,'test',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
    cv2.imshow('Attendance',frame)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()