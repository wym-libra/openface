#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))



import openface



import faceapi
_face_center = faceapi.share_center(
                        os.path.join(fileDir, 'facedb.db3'),
                        os.path.join(fileDir, 'db_face'))



def draw(imgpath, faceInfo):
    img = cv2.imread(imgpath)
    #buf = np.fliplr(np.asarray(img))
    #annotatedFrame = np.copy(buf)
    annotatedFrame = img

    # draw the face area
    area = faceInfo.area
    landmarks = faceInfo.landmarks
    name = faceInfo.name
    bl = (area.left(), area.bottom())
    tr = (area.right(), area.top())
    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                  thickness=3)

    for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
        cv2.circle(
                    annotatedFrame,
                    center=landmarks[p],
                    radius=3,
                    color=(102, 204, 255),
                    thickness=-1)
    cv2.putText(
                annotatedFrame,
                name,
                (area.left(), area.top() - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(152, 255, 204),
                thickness=2)
    cv2.imwrite("images/annotatedFrame.jpg", annotatedFrame)


if __name__ == '__main__':

    #_face_center.addFace("images/test_src.jpg","test")

    #print _face_center.faceList()

    #_face_center.addDir("images/train/")

    facelist = _face_center.faceList()
    print len(facelist)

    face = _face_center.processImg("images/test/andy_7.jpg")

    draw("images/test/andy_7.jpg", face[0])

    print face[0].name

