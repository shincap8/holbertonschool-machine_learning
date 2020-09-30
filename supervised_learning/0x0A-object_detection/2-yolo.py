#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3
algorithm to perform object detection"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Yolo Class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        with open(classes_path, 'r') as f:
            classes_t = f.readlines()
            classes = [x.strip() for x in classes_t]
        self.model = K.models.load_model(model_path)
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sig(self, x):
        """sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """public method to process the output"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sig(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sig(-output[..., 5:]))
        for i in range(len(boxes)):
            gridh = boxes[i].shape[0]
            gridw = boxes[i].shape[1]
            anchor = boxes[i].shape[2]
            t_x = boxes[i][..., 0]
            t_y = boxes[i][..., 1]
            t_w = boxes[i][..., 2]
            t_h = boxes[i][..., 3]
            box = np.zeros((gridh, gridw, anchor))
            indexX = np.arange(gridw).reshape(gridw, 1, 1)
            indexY = np.arange(gridh).reshape(gridh, 1, 1)
            boxX = box + indexX
            boxY = box + indexY
            ntx = self.sig(t_x)
            nty = self.sig(t_y)
            bx = ntx + boxX
            by = nty + boxY
            bx = bx / gridw
            by = by / gridh
            anchorw = self.anchors[i, :, 0]
            anchorh = self.anchors[i, :, 1]
            bw = anchorw * np.exp(t_w)
            bh = anchorh * np.exp(t_h)
            inputw = self.model.input.shape[1].value
            inputh = self.model.input.shape[2].value
            bw = bw / inputw
            bh = bh / inputh
            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][..., 0] = x1 * image_size[1]
            boxes[i][..., 1] = y1 * image_size[0]
            boxes[i][..., 2] = x2 * image_size[1]
            boxes[i][..., 3] = y2 * image_size[0]
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """public method to filter the boxes"""
        box_scores = [x * y for x, y in zip(box_confidences, box_class_probs)]
        box_class_scores = [np.max(x, axis=-1).reshape(-1) for x in box_scores]
        box_class_scores = np.concatenate(box_class_scores)
        box_classes = [np.argmax(x, axis=-1).reshape(-1) for x in box_scores]
        box_classes = np.concatenate(box_classes)
        filtering_mask = box_class_scores >= self.class_t
        list = [np.reshape(x, (-1, 4)) for x in boxes]
        boxes = np.concatenate(list)
        boxes = boxes[filtering_mask]
        scores = box_class_scores[filtering_mask]
        classes = box_classes[filtering_mask]
        return (boxes, classes, scores)
