import flask
import werkzeug
import cv2
from cv2 import resize
import numpy as np
import tensorflow as tf
import math
from scipy.ndimage import interpolation
import cv2
from openai import OpenAI
import base64
import requests

app = flask.Flask(__name__)

ifWide = 2100
ifLong = 1200
counter = 0
global_filename = ''

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    image = cv2.imread(filename, 0)

    rescaled = rescale(image)
    binarized = binarize(rescaled)

    withoutLines = find_hough_lines(binarized, filename)

    if (filename == 'true.jpg'):
        try:
            withoutLines = remove_header(withoutLines)
            inverted_cropped = cv2.bitwise_not(withoutLines)
            cv2.imwrite("androidFlask.jpg", inverted_cropped)
        except:
            return "Couldn't remove header"
    else:
        cv2.imwrite("androidFlask.jpg", image)
    base64_image = encode_image("androidFlask.jpg")

    headers = {
    "Content-Type": "application/json",
    "Authorization": f""
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Return extracted and summed numbers in each column. Response should be GAMER 1: summed_first_column (first_number, second_number, etc.), GAMER 2: summed_second_column etc."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    
    try:
        return response.json()['choices'][0]['message']['content']
    except:
        return "Internal server error"

def remove_header(img):
    length = np.array(img).shape[1]//80

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vertical_detect = cv2.erode(img, vertical_kernel, iterations=2)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=2)

    contours, hierarchy = cv2.findContours(ver_lines, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)        
        if(h < 5 * w and w > 100) :
            cropped = im2[y:y + h, x:x + w]
            boxes.append([x, y, w, h])

            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)


    boxes = sorted(boxes, key=lambda x: x[0], reverse=False)
    if len(boxes) < 2:
        raise IndexError
    else:
        res = im2[max(boxes[0][1], boxes[1][1])+ max(boxes[0][3], boxes[1][3]):im2.shape[0],0:im2.shape[0]]
        return rescale(res)


def rescale(img, ifWide = 2100, ifLong = 1200):
    if img.shape[0]<img.shape[1]:
        base = ifWide
    else: 
        base = ifLong
    img = cv2.resize(img, (base, int(base*img.shape[0]/img.shape[1])), interpolation=cv2.INTER_CUBIC)

    return img


def binarize(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    image = cv2.erode(image, kernel, iterations=7)

    convert_bin, gray = cv2.threshold(image,90,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = 255-gray

    return gray


def find_hough_lines(img, filename):
    im_height = img.shape[0]

    h = cv2.HoughLinesP(img,1,np.pi/160,100,im_height*0.25, im_height/3)

    blank = img.copy()
    blank[:] = 255

    for i in h:
        for x1,y1,x2,y2 in i:
            cv2.line(blank,(x1,y1),(x2,y2),(0,255,0),12)

    img = cv2.imread(filename, 0)
    if img.shape[0]<img.shape[1]:
        base = 2100
    else: 
        base = 1200
    img = cv2.resize(img, (base, int(base*img.shape[0]/img.shape[1])), interpolation=cv2.INTER_CUBIC)
    im_bin, gray = cv2.threshold(img,100,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    xx, im = cv2.threshold(img,100,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    nice = (blank)-(im)

    return nice

app.run(host="127.0.0.1", port=8005, debug=True)