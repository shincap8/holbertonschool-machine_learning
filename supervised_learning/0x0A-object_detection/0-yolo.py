#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3
algorithm to perform object detection"""

import tensorflow.keras as K


class Yolo:
    """Yolo Class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        with open(classes_path) as f:
            classes_t = f.readlines()
        classes = [x.strip() for x in classes_t]
        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
