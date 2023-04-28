import cv2
import numpy as np

def katmanlar(basla, bitir, kk):
    
    colors = []
    sayi = (bitir - basla) // kk # katman kalınlığı seçerek
    
    for i in range(sayi):

        alt = basla + (i * kk)
        ust = basla + ((i + 1) * kk)
        colors.append((np.array([alt, 100, 100]), np.array([ust, 255, 255])))
 
    return colors

basla = 10
bitir = 95
kk = 25
renkler = katmanlar(basla, bitir, kk)


renkler_dizi = {}

for i, renk in enumerate(renkler):
    hangirenk = 'katman{}'.format(i)
    renkler_dizi[hangirenk] = [renk[0].tolist(), renk[1].tolist()]
    print(renkler_dizi, '\n')
    

cap = cv2.VideoCapture('111.mp4')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = {}

    for renk, values in renkler_dizi.items():
        mask = cv2.inRange(hsv, np.array(values[0]), np.array(values[1]))
        masks[renk] = mask



    for color, mask in masks.items():
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 200: 
                x, y, w, h = cv2.boundingRect(contour) #min-max en-boy ekle

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(color, mask)

        

    cv2.imshow('frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


