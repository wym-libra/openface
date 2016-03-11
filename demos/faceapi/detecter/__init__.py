# -*- coding: UTF-8 -*-

"""
@file __init__.py
@brief
    Defines for face detect center.

Created on: 2016/1/14
"""

import os
from abc import ABCMeta, abstractmethod


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class FaceDetected():
    def __init__(self):
        self.img = None
        self.area = None
        self.landmarks = None


class FaceDetector():
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image):
        pass


"""
8888888888                   888
888                          888
888                          888
8888888     8888b.   .d8888b 888888  .d88b.  888d888 888  888
888            "88b d88P"    888    d88""88b 888P"   888  888
888        .d888888 888      888    888  888 888     888  888
888        888  888 Y88b.    Y88b.  Y88..88P 888     Y88b 888
888        "Y888888  "Y8888P  "Y888  "Y88P"  888      "Y88888
                                                          888
                                                     Y8b d88P
                                                      "Y88P"
 """


def make_detector():
    from faceapi.detecter.openface import FaceDetectorOf
    return FaceDetectorOf()
    #from faceapi.detecter.detector_opencv import FaceDetectorOPENCV
    #return FaceDetectorOPENCV()
