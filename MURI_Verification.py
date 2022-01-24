import numpy as np
import cv2
import time
from PIL import Image
import torch
import torchvision
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from win32 import win32gui
import win32ui, win32con, win32api #use pywin32 version 300
import os
import matplotlib.pyplot as plt

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)


    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8   #GPU memory setting
session = tf.Session(config=config)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

while True:
    image_array = grab_screen(region=(0, 0, 2560, 1440)) # screenshot size
    array_to_image = Image.fromarray(image_array, mode='RGB') #arry to image
    img = model(array_to_image)  #yolo detection
    img.save('runs\detect\exp')
    img = cv2.imread('runs\detect\exp\image0.jpg')
    img = cv2.resize(img, (1080, 610))
    cv2.imshow('window',img)
    os.remove('runs\detect\exp\image0.jpg')
    if cv2.waitKey(25) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
       break
