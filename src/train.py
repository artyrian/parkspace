"""
Mask R-CNN
Train on the car top dataset
------------------------------------------------------------
    # Train a new model starting from pre-trained COCO weights
    python3 train.py --dataset=/path/to/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 train.py --dataset=/path/to/dataset --weights=last
"""

import argparse
import os
import sys
import json
import logging
import time

import numpy as np
import skimage.draw


# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'mask_rcnn_coco.h5')

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect car tops.')
    parser.add_argument('--dataset', required=True,
                        metavar='/path/to/dataset/',
                        help='Directory of the car top dataset')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to weights .h5 file or "coco"')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='/path/to/logs/',
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--epochs', default=10,
                        help='Count of iterations learning. Default 10.')
    args = parser.parse_args()

    logging.info('Weights: ', args.weights)
    logging.info('Dataset: ', args.dataset)
    logging.info('Logs: ', args.logs)

    # Configurations
    config = CartopConfig()
    config.display()
    model = modellib.MaskRCNN(
        mode='training',
        config=config,
        model_dir=args.logs
    )
    load_weights_to_model(args.weights, model)

    train(model, args.dataset, args.epochs)


############################################################
#  Configurations
############################################################

def load_weights_to_model(weights, model):
    logging.info('load weights: %s, model: %s', weights, model)

    if weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == 'last':
        weights_path = model.find_last()[1]
    elif weights.lower() == 'imagenet':
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    logging.info('Loading weights ', weights_path)
    if weights.lower() == 'coco':
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=[
                'mrcnn_class_logits',
                'mrcnn_bbox_fc',
                'mrcnn_bbox',
                'mrcnn_mask'
            ]
        )
    else:
        model.load_weights(weights_path, by_name=True)


class CartopConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'cartop'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CartopDataset(utils.Dataset):

    def load_cartop(self, dataset_dir, subset):
        """Load a subset of the car top dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class('cartop', 1, 'cartop')

        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)

        data_json_path = os.path.join(dataset_dir, 'data.json')
        with open(data_json_path, 'r') as f:
            annotations = json.loads(f.read())
            annotations = annotations['_via_img_metadata']
            annotations = list(annotations.values())

        annotations = [a for a in annotations if a.get('regions')]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                'cartop',
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a car top dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != 'cartop':
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'cartop':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset, epochs):
    """Train the model."""
    dataset_train = CartopDataset()
    dataset_train.load_cartop(dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CartopDataset()
    dataset_val.load_cartop(dataset, 'val')
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    logging.info('Training network heads')
    model.train(dataset_train, dataset_val,
                learning_rate=Config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')

    model_path = 'mask_rcnn_' + '.' + str(time.time()) + '.h5'
    model.keras_model.save_weights(os.path.join(ROOT_DIR, model_path))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    main()
