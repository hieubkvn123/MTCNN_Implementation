import cv2
import matplotlib.pyplot as plt 
from mtcnn import MTCNN

test_file = 'test/test11.jpg'
detector = MTCNN(steps_threshold=[0.8,0.9,0.9])

img = cv2.imread(test_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (500, 800))

detector.detect_faces(img)

proposals = detector._proposals[:, 0:4]
for proposal in proposals:
	x1, y1, x2, y2 = [int(x) for x in proposal]
	img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imwrite('test_result.jpg', img)
print('Test result saved into test_result.jpg')
