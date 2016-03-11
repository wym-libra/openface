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



class FaceDetectorOf(FaceDetector):
    def __init__(self):
        super(FaceDetectorOf, self).__init__()
        self._logger = log_center.make_logger(__name__, logging.INFO)

    def detect(self, rgbFrame):
        """
        if isinstance(image, basestring):
            img = Image.open(image)
            self._logger.debug("PIL image: {}".format(str(img)))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise exceptions.LibError("Unknow image type")

        img = _resize(img)
        #buf = np.fliplr(np.asarray(img))
        buf = np.asarray(img)
        im_width, im_height = img.size
        rgbFrame = np.zeros(
                        # (_DEFAULT_IMG_H, _DEFAULT_IMG_W, 3),
                        (im_height, im_width, 3),
                        dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 0]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 2]
        """
        face_box = openfaceutils.align.getAllFaceBoundingBoxes(rgbFrame)
        return face_box
        """
        # face_list = align.getAllFaceBoundingBoxes(rgbFrame)
        face_list = [face_box] if face_box is not None else []

        detected_list = []
        for face_box in face_list:
            landmarks = openfaceutils.align.findLandmarks(rgbFrame, face_box)

            alignedFace = openfaceutils.align.align(
                        openfaceutils.args.imgDim, rgbFrame, face_box,
                        landmarks=landmarks,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue

            face = FaceDetected()
            face.img = alignedFace
            face.area = face_box
            face.landmarks = landmarks

            detected_list.append(face)

        return detected_list
        """