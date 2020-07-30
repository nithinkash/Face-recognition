#!/usr/bin/env python
# coding: utf-8
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["henry"] = img_to_encoding("images/henry.jpg", FRmodel)
database["nithin"] = img_to_encoding("images/nithin.jpg", FRmodel)
database["ben"] = img_to_encoding("images/ben.jpg", FRmodel)

def verify(image_path, identity, database, model):

    encoding = img_to_encoding(image_path, model)

    dist = np.linalg.norm(encoding-database[identity])

    if dist<0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
    return dist, door_open


#verify("images/nithin.jpg", "nithin", database, FRmodel)

verify("images/idk1.jpg", "henry", database, FRmodel)


def who_is_it(image_path, database, model):

    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    
    for (name, db_enc) in database.items():

        dist = np.linalg.norm(encoding-db_enc)
        if dist<min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("Not in the database the distance is "+str(min_dist))
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity


who_is_it("images/nithin.jpg", database, FRmodel)