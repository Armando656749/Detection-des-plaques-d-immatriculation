import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img = cv2.imread("img_char_test.jpg")
img = cv2.resize(img, (1024,512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Ici j'affiches les chiffres et les lettres detectes sur l'image
# print(pytesseract.pytesseract.image_to_string(img))

# himg, wimg,_ = img.shape
# #conf = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_boxes(img)
# print(boxes)

# for boxe in boxes.splitlines():
#     # print(boxe)
#     boxe = boxe.split(' ')
#     print(boxe)
#     x,y,w,h = int(boxe[1]), int(boxe[2]), int(boxe[3]), int(boxe[4])
#     cv2.rectangle(img, (x,himg-y), (w,himg-h), (0,0,255), 2)
#     cv2.putText(img, boxe[0], (x,himg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),2)



# 1. Detection de caracteres

himg, wimg,_ = img.shape
conf = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img, config=conf)
print(boxes)

#print(pytesseract.image_to_boxes(img))
#conf = r'--oem 3 psm 6 outputbase digits'
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#boxes = pytesseract.image_to_boxes(img, config=conf)
for count, boxe in enumerate(boxes.splitlines()):
    #print(boxe)
    if count!=0:
        boxe = boxe.split()
        print(boxe)
        if len(boxe)==12:
            x,y,w,h = int(boxe[6]), int(boxe[7]), int(boxe[8]),int(boxe[9])
            cv2.rectangle(img, (x,y), (w+x,h+y),(0,0,255),2)
            cv2.putText(img, boxe[11], (x,y), cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)



# 2. Detection de chiffres

# conf = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(img, config=conf)
# print(boxes)

# #print(pytesseract.image_to_boxes(img))
# #conf = r'--oem 3 psm 6 outputbase digits'
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #boxes = pytesseract.image_to_boxes(img, config=conf)
# for count, boxe in enumerate(boxes.splitlines()):
#     #print(boxe)
#     if count!=0:
#         boxe = boxe.split()
#         print(boxe)
#         if len(boxe)==12:
#             x,y,w,h = int(boxe[6]), int(boxe[7]), int(boxe[8]),int(boxe[9])
#             cv2.rectangle(img, (x,y), (w+x,h+y),(0,0,255),2)
#             cv2.putText(img, boxe[11], (x,y), cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)


cv2.imshow("result",img)
cv2.waitKey(0)