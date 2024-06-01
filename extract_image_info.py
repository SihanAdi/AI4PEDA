import configparser

from openai import OpenAI
import base64

IMAGE_PATH = 'images/1.PNG'
config = configparser.ConfigParser()
config.read('config.ini')
KEY = config['DEFAULT']['API_KEY']

client = OpenAI(api_key=KEY)
MODEL = 'gpt-4o'


def run_model(model, img_question, base64_img):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 'content': [
                {'type': 'text', 'text': img_question},
                {'type': 'image_url', 'image_url': {
                    'url': f"data:image/PNG;base64,{base64_img}"
                }}
            ]},
            # {'role': 'user', 'content': cont_question}
        ]
    )
    print(response.choices[0].message.content)


def encode_img(path):
    with open(path, 'rb') as img:
        return base64.b64encode(img.read()).decode('utf-8')


if __name__ == '__main__':
    base64_img = encode_img(IMAGE_PATH)
    run_model(MODEL, '请提取图片中的主要模块，并列出图里所有连线的连接关系', base64_img)

