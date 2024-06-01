import base64

IMAGE_PATH = 'images/1.PNG'


def encode_img(path):
    with open(path, 'rb') as img:
        return base64.b64encode(img.read()).decode('utf-8')


base64_img = encode_img(IMAGE_PATH)