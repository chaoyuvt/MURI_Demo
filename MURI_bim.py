import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import argparse
from imagenet_labels import classes
import time
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from win32 import win32gui
import win32ui, win32con, win32api #use pywin32 version 300
import matplotlib.pyplot as plt
import os

IMG_SIZE = 512
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8   #GPU memory setting
session = tf.Session(config=config)

def nothing(x):
    pass

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

print('Iterative Method')
model_name = 'googlenet'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
modely = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = getattr(models, model_name)(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()
model.to(device)


break_loop = False

def nothing(x):
    global break_loop
    break_loop = True

window_adv = 'perturbation'
cv2.namedWindow(window_adv)
cv2.createTrackbar('eps', window_adv, 1, 255, nothing)
#cv2.createTrackbar('alpha', window_adv, 1, 255, nothing)
cv2.createTrackbar('iter', window_adv, 10, 1000, nothing)



while True:

    image_array = grab_screen(region=(0, 0, 2560, 1440)) # screenshot size
    orig = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    img = orig.copy().astype(np.float32)
    #perturbation = np.empty_like(orig)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean)/std
    img = img.transpose(2, 0, 1)


    # prediction before attack
    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
    orig = torch.from_numpy(img).float().to(device).unsqueeze(0)

    out = model(inp)
    pred = np.argmax(out.data.cpu().numpy())

    #if target is not None:
        #pred = target

    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
    eps = cv2.getTrackbarPos('eps', window_adv)
    alpha = 10 #alpha = cv2.getTrackbarPos('alpha', window_adv)
    num_iter = cv2.getTrackbarPos('iter', window_adv)

    print('eps [%d]' %(eps))
    print('Iter [%d]' %(num_iter))
    print('alpha [1]')
    print('-'*20)

    break_loop = False

    for i in range(num_iter):

        if break_loop == False:

            ##############################################################
            out = model(inp)
            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

            loss.backward()

            # this is the method
            perturbation = (alpha/255.0) * torch.sign(inp.grad.data)
            perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
            inp.data = orig + perturbation

            inp.grad.data.zero_()
            ################################################################

            pred_adv = np.argmax(model(inp).data.cpu().numpy())

            print("Iter [%3d/%3d]"
                    %(i, num_iter,))


            # deprocess image
            adv = inp.data.cpu().numpy()[0]
            pert = (adv-img).transpose(1,2,0)
            adv = adv.transpose(1, 2, 0)
            adv = (adv * std) + mean
            adv = adv * 255.0
            adv = adv[..., ::-1] # RGB to BGR
            adv = np.clip(adv, 0, 255).astype(np.uint8)
            pert = pert * 255
            pert = np.clip(pert, 0, 255).astype(np.uint8)



     #yolo detection
    orig = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    img_adv = modely(adv)
    img_ori = modely(orig)
    img_adv.save('runs\detect\exp0')
    img_ori.save('runs\detect\exp1')
    img_adv = cv2.imread('runs\detect\exp0\image0.jpg')
    img_ori = cv2.imread('runs\detect\exp1\image0.jpg')


    # display images
    img_adv = cv2.resize(img_adv, (500, 300))
    img_ori = cv2.resize(img_ori, (500, 300))
    perturbation = cv2.resize(pert, (500, 300))
    cv2.imshow(window_adv, perturbation)
    cv2.imshow('Original Image',img_ori)
    cv2.imshow('AE attack image', img_adv)

    os.remove('runs\detect\exp0\image0.jpg')
    os.remove('runs\detect\exp1\image0.jpg')
    key = cv2.waitKey(500) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('img_adv.png', adv)
        cv2.imwrite('perturbation.png', perturbation)
print()
cv2.destroyAllWindows()
os._exit(0)
