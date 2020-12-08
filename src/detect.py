import argparse
import cv2
import glob
import logging
import os

import numpy as np

from mrcnn import config
from mrcnn import visualize
from mrcnn.model import MaskRCNN

MODEL_DIR = 'logs'

RED_COLOR = (0, 0, 255)  # BGR
WHITE_COLOR = (255, 255, 255)


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(config.Config):
    NAME = 'mask_trained_model_config'
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.7


class CarDetection:

    def __init__(self, box, mask, score):
        self.box = box
        self.mask = mask
        self.score = score


class CarTopDetector:

    def __init__(self, weights_path):
        model = MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=MaskRCNNConfig())
        model.load_weights(weights_path, by_name=True)

        self.model = model

    def detect(self, frame):
        rgb_image = frame[:, :, ::-1]
        results = self.model.detect([rgb_image], verbose=0)  # todo: use frames list
        r = results[0]  # r data: rois, class_ids, scores, masks

        result = []
        for box, mask, score in zip(r['rois'], r['masks'], r['scores']):
            result.append(CarDetection(box, mask, score))

        return result


def main():
    logging.basicConfig(format='%(asctime)-15s %(levelname)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect car tops.')
    parser.add_argument('--data', required=True,
                        metavar='/path/to/data/',
                        help='Directory with images to detect')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to weights .h5 file')
    parser.add_argument('--out', required=True,
                        metavar='/path/to/out',
                        help='Directory for out images with detected objects')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    detector = CarTopDetector(weights_path=args.weights)

    images_list = glob.glob(os.path.join(args.data, '*.jpg'))

    for img_path in images_list:
        logging.info('load frame %s', img_path)
        frame = cv2.imread(img_path)
        result = detector.detect(frame)
        for detection in result:
            y1, x1, y2, x2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED_COLOR, 1)
            cv2.putText(frame, text='P', org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=WHITE_COLOR)

        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.out, frame_name), frame)


if __name__ == '__main__':
    main()
