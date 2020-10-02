#!/usr/bin/env python3
"""class Yolo that uses the Yolo v3
algorithm to perform object detection"""

import glob
import cv2 as cv
import numpy as np
import os
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
            box_class_probs.append(self.sig(output[..., 5:]))
        for i in range(len(boxes)):
            gridh = boxes[i].shape[0]
            gridw = boxes[i].shape[1]
            anchor = boxes[i].shape[2]
            t_x = boxes[i][..., 0]
            t_y = boxes[i][..., 1]
            t_w = boxes[i][..., 2]
            t_h = boxes[i][..., 3]
            box = np.zeros((gridh, gridw, anchor))
            indexX = np.arange(gridw).reshape(1, gridw, 1)
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
        box_classes = [np.argmax(x, axis=-1).reshape(-1)for x in box_scores]
        box_classes = np.concatenate(box_classes)
        filtering_mask = box_class_scores >= self.class_t
        list = [np.reshape(x, (-1, 4)) for x in boxes]
        boxes = np.concatenate(list)
        boxes = boxes[filtering_mask]
        scores = box_class_scores[filtering_mask]
        classes = box_classes[filtering_mask]
        return (boxes, classes, scores)

    def iou(self, boxA, boxB):
        """this is the iou of the box against all other boxes"""
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """public method non max supression"""
        sort_indexes = np.lexsort((-box_scores, box_classes))
        box_predictions = np.array([filtered_boxes[x] for x in sort_indexes])
        p_box_classes = np.array([box_classes[x] for x in sort_indexes])
        predicted_box_scores = np.array([box_scores[x] for x in sort_indexes])
        _, count_class = np.unique(p_box_classes, return_counts=True)

        index_collect = 0
        i = 0
        for n in count_class:
            while i < index_collect + n:
                j = i + 1
                while j < index_collect + n:
                    iou = self.iou(box_predictions[i], box_predictions[j])
                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions, j, axis=0)
                        p_box_classes = np.delete(p_box_classes, j, axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         j, axis=0)
                        n = n - 1
                    else:
                        j = j + 1
                i = i + 1
            index_collect = index_collect + n
        return (box_predictions, p_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """method to load image"""
        image_paths = glob.glob(folder_path + '/*.jpg')
        images = [cv.imread(x) for x in image_paths]
        return (images, image_paths)

    def preprocess_images(self, images):
        """public method to preprocess images"""
        inputw = self.model.input.shape[1].value
        inputh = self.model.input.shape[2].value
        pimages = []
        image_shapes = []
        for i in range(len(images)):
            newX = images[i].shape[0]
            newY = images[i].shape[1]
            image_shapes.append((newX, newY))
            resize = cv.resize(images[i], (inputw, inputh),
                               interpolation=cv.INTER_CUBIC)
            resize = resize / 255
            pimages.append(resize)
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """public method to show boxes"""
        for i in range(len(boxes)):
            startX = int(boxes[i, 0])
            startY = int(boxes[i, 1])
            endX = int(boxes[i, 2])
            endY = int(boxes[i, 3])
            cv.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
            name = self.class_names[box_classes[i]]
            text = "{} {:.2f}".format(name, box_scores[i])
            cv.putText(image, text, (startX, startY - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                       1, cv.LINE_AA)
            cv.imshow(file_name, image)
            k = cv.waitKey(0)
            if k == ord('s'):
                if not os.path.isdir('detections'):
                    os.mkdir('detections')
                else:
                    None
                os.chdir('detections')
                cv.inwrite(file_name, image)
                os.chdir('detections')
                cv.imwrite(file_name, image)
                os.chdir('../')
            cv.destroyAllWindows()
