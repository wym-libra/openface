# -*- coding: UTF-8 -*-

"""
@file openface.py
@brief
    Implement of img eigener in eigen_center.

Created on: 2016/1/14
"""

import numpy as np
import logging
from PIL import Image

import openface

from faceapi import openfaceutils
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.detecter import FaceDetector
from faceapi.detecter import FaceDetected




class FaceDetectorOf(FaceDetector):
    def __init__(self):
        super(FaceDetectorOf, self).__init__()
        self._logger = log_center.make_logger(__name__, logging.INFO)

    
