# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""
import cv2
import os
import numpy as np
import logging

import dlib

import openface

from faceapi import openfaceutils
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.detecter import FaceDetector


class FaceDetectorOPENCV(FaceDetector):
    def __init__(self):
        super(FaceDetectorOPENCV, self).__init__()
        self._logger = log_center.make_logger(__name__, logging.INFO)
        # Create the haar cascade
        self.faceCascade = cv2.CascadeClassifier(
                         os.path.join(
                                      openfaceutils.opencvModelDir,
                                      "haarcascade_frontalface_alt.xml")) #1

    def detect(self, image):

        # Read the image
        #image = cv2.imread(imagePath)#2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#3
        cv2.equalizeHist(gray, gray)#灰度图像进行直方图等距化 wym
        # Detect faces in the image
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(20,20),
            flags = cv2.CASCADE_SCALE_IMAGE
        ) #4

        facelist = []
        for (x, y, w, h) in faces:

            rec = dlib.rectangle(long(x), long(y), long(x+w), long(y+h))
            facelist.append(rec)


        return facelist
