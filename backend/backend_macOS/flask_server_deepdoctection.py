import flask
import werkzeug
import cv2
from cv2 import resize
import numpy as np
from scipy.ndimage import interpolation
import cv2


import cv2
from pathlib import Path
from matplotlib import pyplot as plt

import deepdoctection as dd

app = flask.Flask(__name__)

ifWide = 2100
ifLong = 1200
counter = 0
global_filename = ''


@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    image = cv2.imread(filename, 0)

    rescaled = rescale(image)
    binarized = binarize(rescaled)

    cv2.imwrite("data/androidFlask.jpg", image)

    analyzer = dd.get_dd_analyzer(config_overwrite=["LANGUAGE='eng'"])
    path = Path.cwd() / "deepdoctection/data"
    print(path)
    df = analyzer.analyze(path=path)
    df.reset_state() 
    doc=iter(df)
    page = next(doc)
    image = page.viz()
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)

    plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0) 
    print(page.tables[0].csv)
    print(len(page.tables))
    print(page.text)

    table = page.tables[0]

    gamers_results = np.zeros(table.shape[1])

    np_array = np.array(table)
    for col in range(np_array.shape[1]): 
        for row in range(np_array.shape[0]): 
            value = np_array[row, col] 
            if value.strip() == '': 
                value = 0
            else: 
                value = float(value) 
            gamers_results[col] += value

    return format_results(gamers_results)

def format_results(arr):
    length = len(arr)
    arr = [str(x) for x in arr]
    st = ''
    index = 1
    for i in arr:
        st = st + 'GAMER ' + str(index) + ' : ' + i + ' '
        index = index + 1
    print(st)
    return st
    
def divide(projection_array, img_nice):
    for i in range(len(projection_array)):
        if(projection_array[i]<5):
            projection_array[i]=0

    if(img_nice.shape[0] < img_nice.shape[1]):
        x = 2100
    else:
        x = 1200

    img_do_podzialu = img_nice.copy()


    for i in range(x):
        if (projection_array[i]==0):
            img_do_podzialu[:, i] = 0
        else:
            img_do_podzialu[:, i] = 255
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ret, thresh1 = cv2.threshold(img_do_podzialu, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    img = cv2.erode(img_do_podzialu, kernel, iterations=2)
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 6)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
    if img.shape[0]<img.shape[1]:
        base = 2100
    else: 
        base = 1200

    im2 = img_nice.copy()

    cropped_columns = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 90 and h > 35) and w < 11 * h:
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped = im2[y:y + h, x:x + w]
            inverted_cropped = cv2.bitwise_not(cropped)
            cropped_columns.append(inverted_cropped)
    
    return cropped_columns
    
def horizontal_projection(img):
    nice = img.copy()

    gray = cv2.medianBlur(nice,5)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h > 10:    
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    height, width = thresh.shape
    vertical_px = np.sum(thresh, axis=0)
    normalize = vertical_px/255
    blankImage = np.zeros_like(thresh)
    for idx, value in enumerate(normalize):
        cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)


    if(img.shape[0] < img.shape[1]):
        kon = np.zeros(2100)
    else:
        kon = np.zeros(1200)
        
    for idx, value in enumerate(normalize):
        cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)
        kon[idx]+=value

    return kon  

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

app.run(host="127.0.0.1", port=8004, debug=True)