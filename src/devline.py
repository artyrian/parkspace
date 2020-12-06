import argparse
import logging
import os
import requests
import time

from urllib.parse import urljoin

CAMERA_IMAGE_TEMPLATE_API = '/cameras/{num}/image'

ACCEPT_TYPE_JSON = 'application/json'
ACCEPT_TYPE_IMAGE = 'application/jpeg'


def main():
    logging.basicConfig(format='%(asctime)-15s %(levelname)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Connect and downlod pics fromm devline cams server.')
    parser.add_argument('--url', required=True, metavar='cam.example.ru',
                        help='URL or IP to server')
    parser.add_argument('--port', required=True, metavar='1234',
                        help='server port to connect')
    parser.add_argument('--user', required=True, metavar='display',
                        help='User for devline server API')
    parser.add_argument('--password', required=True, metavar='password@#1!',
                        help='Password for user devline server API')
    parser.add_argument('--dir', required=True, metavar='/path/to/dir',
                        help='Directory to save screens from cameras')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    img_suffix = str(round(time.time()))  # use timestamp to uniq names on different runs script
    logging.info('Dir to save images: %s, image suffix: %s', args.dir, img_suffix)

    client = DevLineCameraApi(args.url, args.port, args.user, args.password)
    cameras = client.cameras_list()
    logging.info('cameras list: %d items', len(cameras))

    for cam in cameras:
        logging.info('camera: %s', cam['name'])
        path_to_img = _path_img(args.dir, cam['name'], img_suffix)
        client.screen_camera(cam, path_to_img)


def _path_img(dir_to, cam_name, suffix):
    img_name = 'img_{name}_{suffix}.jpg'.format(
        name=cam_name.replace('/', '__'),
        suffix=suffix
    )
    return os.path.join(dir_to, img_name)


class DevLineCameraApi:

    def __init__(self,
                 url,
                 port,
                 user,
                 password
                 ):
        self.server = '{}:{}'.format(url, port)
        self.user = user
        self.password = password

    def _get(self, api_handle, **kwargs):
        url = urljoin(self.server, api_handle)
        headers = {}
        if kwargs.get('headers'):
            headers.update(kwargs['headers'])

        resp = requests.get(
            url,
            auth=(self.user, self.password),
            headers=headers,
        )
        resp.raise_for_status()
        return resp

    def cameras_list(self):
        resp = self._get(
            '/cameras',
            headers={'Accept': 'application/json'}
        )
        return resp.json()

    def _camera_image(self, image_uri):
        resp = self._get(
            image_uri,
            headers={'Content-Type': 'image/jpeg'}
        )
        return resp

    def screen_camera(self, cam, path_img):
        resp = self._camera_image(cam['image-uri'])
        if resp.content:
            self._to_file(resp.content, path_img)
        else:
            logging.warning('Empty data from cam %s', cam['name'])

    @staticmethod
    def _to_file(content, path):
        with open(path, 'wb') as out_file:
            out_file.write(content)


if __name__ == '__main__':
    main()
