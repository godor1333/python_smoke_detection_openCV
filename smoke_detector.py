
import cv2 as cv

#параметры HSV для детектирования дыма. Выбирались изходя из статьи и дальнейшего исправления для лучшей точности
max_value = 255
max_value_H = 360 // 2
low_H = 151
low_S = 0
low_V = 71
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'



cap = cv.VideoCapture("./input.avi")


#Метрика для проверки на сколько один ббокс входит в другой
def IoU(boxA, boxB):

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


#параметры для сохранения видео
fourcc = cv.VideoWriter_fourcc('X','V','I','D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv.VideoWriter('output.mp4', fourcc, 20,(frame_width,frame_height),True )

#статическая разметка дверей
# x_fr = 0
# y_fr = 130
static_door = [[5, 130, 35, 330], [40, 130, 70, 330],
               [75, 130, 105, 330], [110, 130, 140, 330],
               [145, 130, 175, 330], [180, 130, 210, 330],
               [215, 130, 245, 330], [250, 130, 280, 330],
               [285, 130, 315, 330], [320, 130, 350, 330],
               [355, 130, 385, 330], [390, 130, 420, 330],
               [425, 130, 455, 330], [460, 130, 490, 330],
               [495, 130, 525, 330], [530, 130, 560, 330],
               [565, 130, 595, 330], [600, 130, 630, 330],
               [635, 130, 665, 330], [670, 130, 700, 330],
               [705, 130, 735, 330], [740, 130, 770, 330],
               [775, 130, 805, 330], [810, 130, 840, 330],
               [845, 130, 875, 330], [880, 130, 910, 330],
               [915, 130, 945, 330], [950, 130, 980, 330], [985, 130, 1015, 330]]
#читаем покадрово видео
while True:

    ret, frame = cap.read()
    if frame is None:
        break

    # Переводим в формат HSV
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Фильтруем по нашим параметрам
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    #Накладываем блюр с параметром ядра 1 на 1
    blur = cv.GaussianBlur(frame_threshold, (1, 1), 3)
    #Фильтруем изображение чтобы получить бинарную маску
    _, thresh = cv.threshold(blur, 25, 180, cv.THRESH_BINARY)
    #расширение и получение контуров
    dilated = cv.dilate(thresh, None, iterations=3)
    сontours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



    #проходим по контурам, собираем ббоксы, фильтруем их размеры(чтобы убрать шум)
    for contour in сontours:
        (x, y, w, h) = cv.boundingRect(contour)
        print(contour)
        if cv.contourArea(contour) > 350 and cv.contourArea(contour) < 2800:
            #проверяем на сколько ббоксы детектора дыма совпадают с дверью
            for door in static_door:
                # если параметр большой то дыма много
                if IoU(door,(x,y,x+w,h+y)) > 0.3:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                 1)  # получение прямоугольника из точек кортежа
                    cv.putText(frame, "{}".format("smoke"), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1,
                               cv.LINE_AA)
                    # если параметр маленький то дыма мало или вообще нет
                elif IoU(door,(x,y,x+w,h+y)) > 0.05:

                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    cv.putText(frame, "{}".format("smoke"), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1,
                    cv.LINE_AA)

    ####
    out.write(frame)
    cv.imshow(window_capture_name, frame)


    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()