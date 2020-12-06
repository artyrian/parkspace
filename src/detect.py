import glob
import os
import numpy as np
import cv2

import mrcnn2.config
import mrcnn2.visualize
from mrcnn2.model import MaskRCNN
from pathlib import Path

ROOT_DIR = Path('.')
IMAGES_DIR = os.path.join(ROOT_DIR, 'dataset', 'data')
OUT_IMAGES_DIR = os.path.join(ROOT_DIR, 'out')
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'mask_rcnn_coco.h5')


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn2.config.Config):
    NAME = 'coco_pretrained_model_config'
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        print('class id:', class_ids[i])
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


def get_path_images():
    return glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))


def main():
    model = MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Location of parking spaces
    parked_car_boxes = None

    images_list = get_path_images()
    for img_path in images_list:
        frame = cv2.imread(img_path)
        rgb_image = frame[:, :, ::-1]
        results = model.detect([rgb_image], verbose=0)

        # Mask R-CNN assumes we are running detection on multiple images.
        # We only passed in one image to detect, so only grab the first result.
        r = results[0]
        print('keys:', r.keys())

        # The r variable will now have the results of detection:
        # - r['rois'] are the bounding box of each detected object
        # - r['class_ids'] are the class id (type) of each detected object
        # - r['scores'] are the confidence scores for each detection
        # - r['masks'] are the object masks for each detected object (which gives you the object outline)

        # car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        index = 0
        for box in r['rois']:
            class_id = r['class_ids'][index]
            print('detected id:', class_id)
            y1, x1, y2, x2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame,
                        text='id: {}'.format(class_id),
                        org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.5,
                        color=(255, 255, 255))

            index += 1

        # mrcnn.visualize.display_images([frame])
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(OUT_IMAGES_DIR, frame_name), frame)


if __name__ == '__main__':
    main()
