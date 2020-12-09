import argparse
import logging
import time
import traceback

import cv2
import numpy as np

import detect
import devline

IDLE_DETECTING_LOOP_SEC = 10


def main():
    logging.basicConfig(format='%(asctime)-15s %(levelname)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Live detection parking space')
    devline.add_parsing_args(parser)
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to weights .h5 file')
    args = parser.parse_args()

    client = devline.DevLineCameraApi(args.url, args.port, args.user, args.password)
    detector = detect.CarTopDetector(args.weights)

    cameras = client.cameras_list()
    while True:
        safe_detect(client, cameras, detector)
        time.sleep(IDLE_DETECTING_LOOP_SEC)


def safe_detect(client, cameras, detector):
    for cam in cameras:
        name = cam['name']

        try:
            img_data = client.camera_image(cam)
            if not img_data:
                logging.warning('not found image on cam %s', name)
                continue

            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), 1)
            cv2.imwrite('dataset/live/in/{}.jpg'.format(name.replace('/', '__')), img)

            logging.info('start detecting...')
            detections = detector.detect(img)
            logging.info('mark result')
            detect.mark(img, detections)
            logging.info('detected cars: %d on zone %s', len(detections), cam['name'])
            cv2.imwrite('dataset/live/out/{}.jpg'.format(name.replace('/', '__')), img)
        except Exception as e:
            traceback.print_exc()
            logging.error('Failed on process img on cam %s', cam['name'], e)


if __name__ == '__main__':
    main()
