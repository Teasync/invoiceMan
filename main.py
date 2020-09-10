import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
import os
import pytesseract
from pytesseract import Output


def show_img(img, name='image', h=800, w=800):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, h, w)
    cv2.waitKey(0)


def draw_boxes(img, d):
    n_boxes = len(d['level'])
    for i in range(n_boxes - 1):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w//2, y + h//2), (0, 0, 255), 2)
    return img


def find_horz(el: int, d: dict) -> list:
    res = []
    (top, height) = (d['top'][el], d['height'][el])
    for i, n in enumerate(d['top']):
        if top - height // 2 < n < top + height * 1.5:
            res.append(i)
    return res


pdf_fname = 'ex.pdf'

pages = convert_from_path(pdf_fname, 500)

cv2_imgs = []

for page in pages:
    cv2_imgs.append(np.array(page))

# show_img(cv2_imgs[0])


# img = cv2.imread('invoice1.png', 0)
# cv2.imshow('gray', img)
# cv2.waitKey(0)
# ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# kernel = np.ones((3, 3), np.uint8)
# ikmg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

potential_totals = []

for cv2_img in cv2_imgs:
    ret, img = cv2.threshold(cv2_img, 180, 255, cv2.THRESH_BINARY)
    # show_img(img)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # img = draw_boxes(img, d)
    # show_img(img)
    # extracted_text = pytesseract.image_to_string(img)
    # print(extracted_text)
    for i, n in enumerate(d['text']):
        if 'total' in n.lower().strip():
            potential_totals.extend(find_horz(i))
    print('hi')

# ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
# cv2.imshow('gray', img)
# cv2.waitKey(0)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes - 1):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
print('hi')

extracted_text = pytesseract.image_to_string(img)
print(extracted_text)