from imageio import imread, imsave
import numpy as np
import cv2 as cv
import Face_Recognition_v3a as face

image=cv.imread('images/00001.jpg')
image=cv.resize(image, (96, 96))
imsave('/home/nithin/Triplet_loss/images/subbukrishna.jpg', image)

