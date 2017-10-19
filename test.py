import numpy as np
import cv2
oncoming_cascade = cv2.CascadeClassifier('/home/tomasz/Desktop/opencv-tests/headlight-classifier/cascade.xml')
# face_cascade = cv2.CascadeClassifier('/home/tomasz/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/home/tomasz/opencv-3.3.0/data/haarcascades/haarcascade_eye.xml')

photos = [
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m02s111.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m12s677.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m20s414.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m25s981.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m36s869.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m44s271.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m46s546.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-19h11m54s389.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-21h53m04s657.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-21h54m24s516.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-18-21h54m34s118.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m11s210.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m12s523.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m13s402.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m26s688.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m29s357.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m43s049.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m44s364.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m48s247.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m53s325.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m56s529.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h36m58s840.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m22s414.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m23s773.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m33s815.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m42s162.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m53s562.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h37m59s601.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m00s520.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m00s911.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m09s327.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m10s197.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m11s471.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m13s113.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m23s123.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m30s843.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m33s321.png',
    '/home/tomasz/Pictures/vlcsnap-2017-10-19-17h38m37s117.png'
]

for path in photos:
    img = cv2.imread(path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vehicles = oncoming_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in vehicles:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
