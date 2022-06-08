from __future__ import unicode_literals
from flask import Flask, request, abort, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import requests
import json
import configparser
import os
import random
import string
from urllib import parse
app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])


config = configparser.ConfigParser()
config.read('config.ini')

line_bot_api = LineBotApi(config.get('line-bot', 'channel_access_token'))
handler = WebhookHandler(config.get('line-bot', 'channel_secret'))
my_line_id = config.get('line-bot', 'my_line_id')
end_point = config.get('line-bot', 'end_point')
line_login_id = config.get('line-bot', 'line_login_id')
line_login_secret = config.get('line-bot', 'line_login_secret')
my_phone = config.get('line-bot', 'my_phone')
HEADER = {
    'Content-type': 'application/json',
    'Authorization': F'Bearer {config.get("line-bot", "channel_access_token")}'
}
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image


def class_convert(classess):
    pred=[]
    for i in classess:
        if i ==0:
            pred.append('Cardboard')
        elif i==1:
            pred.append('metal_aluminum')
        elif i==2:
            pred.append('paper_container')
        elif i==3:
            pred.append('Plastic')
        elif i==4:
            pred.append('Trash')
    return pred

model = tf.keras.models.load_model('./model_v3.h5/')

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return 'ok'
    body = request.json
    events = body["events"]
    print(body)
    if "replyToken" in events[0]:
        payload = dict()
        replyToken = events[0]["replyToken"]
        payload["replyToken"] = replyToken
        if events[0]["type"] == "message":
            if events[0]["message"]["type"] == "text":
                text = events[0]["message"]["text"]

                if text == "今日確診人數":
                    payload["messages"] = [
                            {
                                "type": "text",
                                "text": getTodayCovid19Message()
                            }
                        ]
                elif text == "開始辨識":
                    payload["messages"] = [
                        {
                            "type": "text",
                            "text": "開啟相機或點選圖庫照片，以開始辨識",
                            "quickReply": {
                                "items": [
                                    {
                                        "type": "action",
                                        "action": {
                                           "type": "camera",
                                            "label": "開啟相機"
                                        }
                                    },
                                    {
                                        "type": "action",
                                        "action": {
                                            "type": "cameraRoll",
                                            "label": "開啟圖庫",
                                        }
                                    }
                                ]
                            }
                        }

                    ]
                else:
                    payload["messages"] = [
                            {
                                "type": "text",
                                "text": "請輸入正確指令"
                            }
                        ]
                replyMessage(payload)
            elif events[0]["message"]["type"] == "image":
                image_name = ''.join(random.choice(string.ascii_letters + string.digits) for x in range(4))
                image_content = line_bot_api.get_message_content(events[0]["message"]["id"])
                image_name = image_name.upper() + '.jpg'
                path = './static/trash/' + image_name
                with open(path, 'wb') as fd:
                    for chunk in image_content.iter_content():
                        fd.write(chunk)

                def classify_image(my_image):
                    global result
                    custom_image = image.load_img(my_image, target_size=(224, 224))
                    img_array = image.img_to_array(custom_image)
                    processed_img = keras.applications.efficientnet_v2.preprocess_input(img_array).astype(np.float32)
                    swapped = np.moveaxis(processed_img, 0, 1)
                    arr4d = np.expand_dims(swapped, 0)
                    new_prediction = class_convert(np.argmax(model.predict(arr4d), axis=-1))  # 用class_convert進行文字分類
                    print('Your item is: ', new_prediction[0])
                    result = str(new_prediction[0])

                classify_image(path)
                payload["messages"] = [
                        {
                            "type": "text",
                            "text": F"判斷為:{result}"
                        }
                    ]
                replyMessage(payload)



    return 'OK'




def getTodayCovid19Message():
    response = requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4001&limited=TWN", headers=HEADER)
    data = response.json()[0]
    date = data["a04"]
    total_count = data["a05"]
    count = data["a06"]
    return F"日期：{date}, 人數：{count}, 確診總人數：{total_count}"

def replyMessage(payload):
    response = requests.post("https://api.line.me/v2/bot/message/reply", headers=HEADER, data=json.dumps(payload))
    return 'OK'

if __name__ == "__main__":
    app.debug = True
    app.run()
