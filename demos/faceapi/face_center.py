# -*- coding: UTF-8 -*-

"""
@file face_center.py
@brief
    Implement of FaceCenter.

Created on: 2016/1/21
"""

import os
import glob
import logging
import time
# import imagehash
from PIL import Image
import numpy as np

import faceapi
from faceapi import FaceCenter
from faceapi.utils import log_center
from faceapi import openfaceutils
import openface
import time
"""
8888888b.            .d888 d8b
888  "Y88b          d88P"  Y8P
888    888          888
888    888  .d88b.  888888 888 88888b.   .d88b.  .d8888b
888    888 d8P  Y8b 888    888 888 "88b d8P  Y8b 88K
888    888 88888888 888    888 888  888 88888888 "Y8888b.
888  .d88P Y8b.     888    888 888  888 Y8b.          X88
8888888P"   "Y8888  888    888 888  888  "Y8888   88888P'
"""
_DEFAULT_IMG_W = 400
_DEFAULT_IMG_H = 300

_IMG_RESIZE_BASE = 512.0


def _resize(img):
    im_width, im_height = img.size

    if max(im_width, im_height) < _IMG_RESIZE_BASE:
        return img

    if im_width >= im_height:
        # resize base on width
        ratio = _IMG_RESIZE_BASE / im_width
        im_height = int(ratio * im_height)
        im_width = int(_IMG_RESIZE_BASE)
    else:
        # resize base on height
        ratio = _IMG_RESIZE_BASE / im_height
        im_width = int(ratio * im_width)
        im_height = int(_IMG_RESIZE_BASE)

    img = img.resize((im_width, im_height), Image.BILINEAR)
    return img

def getAlignedFace(rgbFrame, face_box):
        
    landmarks = openfaceutils.align.findLandmarks(rgbFrame, face_box)

    alignedFace = openfaceutils.align.align(
                openfaceutils.args.imgDim, rgbFrame, face_box,
                landmarks=landmarks,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    return landmarks,alignedFace

def getRGBImg(imgPath):

    img = Image.open(imgPath)

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

    return rgbFrame


class FaceCenterOf(FaceCenter):
    def __init__(self, db_path, trained_face_dir):
        super(FaceCenterOf, self).__init__(db_path, trained_face_dir)
        self._log = log_center.make_logger(__name__, logging.DEBUG)

        self._trained_face_dir = trained_face_dir
        self._face_db = faceapi.database.make_db_manager(db_path)
        self._face_detector = faceapi.detecter.make_detector()
        self._face_eigener = faceapi.eigener.make_eigener()
        self._face_classifier = faceapi.classifier.make_classifier(db_path)

    def faceList(self):
        list = []
        for info in self._face_db.dbList():
            face_info = faceapi.FaceInfo(
                                info['hash'].encode('ascii', 'ignore'),
                                info['name'],
                                [float(x) for x in info['eigen'].split(',')],
                                info['img_path'].encode('ascii', 'ignore'),
                                info['class_id'])
            list.append(face_info)
        return list

    def addFace(self, image, name):
        
        rgbFrame = getRGBImg(image)

        bbs = self._face_detector.detect(rgbFrame)

        trained_list = []
        for face_box in bbs:
            alignedFace = getAlignedFace(rgbFrame, face_box)

            phash, rep = self._face_eigener.eigenValue(alignedFace)

            identity = self._toIdentity(name)
            if identity is None:
                people_list = self._face_db.distinct_search(
                                            ['name', 'class_id'], 'class_id')

                identity = len(people_list)

            face_img = os.path.join(
                    self._trained_face_dir, "{}_{}.jpg".format(name, phash))
            Image.fromarray(face.img).save(face_img)
            record = faceapi.FaceInfo(
                            phash, name, rep, face_img, identity)
            self._face_db.addList([record])
            trained_list.append(record)
            # content = [str(x) for x in face.img.flatten()]

        #self._face_classifier.updateDB()
        return trained_list

    def predict(self, image, callback=None):
        bbs = self._face_detector.detect(image)
        hit_cnt = 0
        for face in bbs:
            hit = self._face_classifier.predict(face.img)
            if hit is None:
                continue
            hit_cnt += 1
            if callback is not None:
                callback(hit.class_id, face.area, face.landmarks)

        return hit_cnt

    def processImg(self, image):
        start = time.time()
        rgbFrame = getRGBImg(image)
        print("  + getRGBImg took {} seconds.".format(time.time() - start))

        start = time.time()
        bbs = self._face_detector.detect(rgbFrame)
        print("  + face_detector took {} seconds.".format(time.time() - start))

        faceList = []
        for face_box in bbs:
            start = time.time()
            landmarks,alignedFace = getAlignedFace(rgbFrame, face_box)
            print("  + getAlignedFace took {} seconds.".format(time.time() - start))

            start = time.time()
            hit = self._face_classifier.predict(alignedFace)
            print("  + face_classifier took {} seconds.".format(time.time() - start))
            
            if hit is None:
                continue
            face = faceapi.Face(hit.name, face_box, landmarks)
            faceList.append(face)

        return faceList

    def addDir(self, dir_path):
        print "addDir"
        if not os.path.isdir(dir_path):
            self._log.error('Not a dir, do nothing.\n({})'.format(dir_path))
            return
        print dir_path
        print next(os.walk(dir_path))
        img_names = next(os.walk(dir_path))[2]
        print img_names
        for img_name in img_names:
            path = os.path.join(dir_path, img_name)
            person_name = img_name[:-6]

            self.addFace(path, person_name)

    def _toIdentity(self, name):
        db_name_map = self._face_db.distinct_search(
                                            ['name', 'class_id'], 'class_id')

        if len(db_name_map) == 0:
            return None

        check_ret = [
                    (name_dic['name'], name_dic['class_id'])
                    for name_dic in db_name_map
                    if name_dic['name'] == name]

        if len(check_ret) == 0:
            return None

        class_id = (check_ret[0])[1]

        return class_id
