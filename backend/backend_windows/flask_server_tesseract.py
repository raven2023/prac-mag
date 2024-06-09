import flask
import werkzeug
import cv2
from cv2 import resize
import numpy as np
import tensorflow as tf
import math
from scipy.ndimage import interpolation
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

    withoutLines = find_hough_lines(binarized, filename)
    if (filename == 'true.jpg'):
        try:
            withoutLines = remove_header(withoutLines)
        except:
            return "Couldn't remove header"

    projection_array = horizontal_projection(withoutLines.copy())
    columns = divide(projection_array, withoutLines)

    rows = []
    gamers_results = []
    final_table = []
    for column in columns:
        projection_row_arrawy = vertical_projection(column.copy())
        rows = divide_rows(projection_row_arrawy, column)
        for row in rows:
            print("-----------------------------------------------")
            number = find_number(row)
            gamers_results.append(number)
        final_table.append(sum(gamers_results))
        gamers_results = []
    final_table = final_table[::-1]
    return format_results(final_table)


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
            print(x, y, w, h) 
            boxes.append([x, y, w, h])

            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)

    boxes = sorted(boxes, key=lambda x: x[0], reverse=False)
    if len(boxes) < 2:
        raise IndexError
    else:
        res = im2[max(boxes[0][1], boxes[1][1])+ max(boxes[0][3], boxes[1][3]):im2.shape[0],0:im2.shape[0]]
        return rescale(res)


def table_to_number(arr):
    print(arr)
    print("aaa", arr)
    res = sum(d * 10**i for i, d in enumerate(arr[::-1]))
    return res

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

def vertical_projection(img):
    if(img.shape[0]>img.shape[1]):
        nice = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
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

    kon = np.zeros(img.shape[0])

        
    for idx, value in enumerate(normalize):
        cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)
        kon[idx]+=value

    return kon  


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
            cropped_columns.append(cropped)
    
    return cropped_columns

def divide_rows(projection_array, img_nice):
    for i in range(len(projection_array)):
        if(projection_array[i]<2):
            projection_array[i]=0

    x = img_nice.shape[0]

    img_do_podzialu = img_nice.copy()


    for i in range(x):
        if (projection_array[i]==0):
            img_do_podzialu[i, :] = 0
        else:
            img_do_podzialu[i, :] = 255

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ret, thresh1 = cv2.threshold(img_do_podzialu, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    img = cv2.erode(img_do_podzialu, kernel, iterations=2)
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 3)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)

    im2 = img_nice.copy()

    cropped_columns = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 60 and h > 35) and w < 11 * h:
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped = im2[y:y + h, x:x + w]
            cropped_columns.append(cropped)
    return cropped_columns

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)

def moments(image):
        c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] 
        totalImage = np.sum(image)
        m0 = np.sum(c0*image)/totalImage 
        m1 = np.sum(c1*image)/totalImage 
        m00 = np.sum((c0-m0)**2*image)/totalImage 
        m11 = np.sum((c1-m1)**2*image)/totalImage 
        m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage 
        mu_vector = np.array([m0,m1]) 
        covariance_matrix = np.array([[m00,m01],[m01,m11]]) 
        return mu_vector, covariance_matrix

def deskew(image):
        c,v = moments(image)
        alpha = v[0,1]/v[0,0]
        affine = np.array([[1,0],[alpha,1]])
        ocenter = np.array(image.shape)/2.0
        offset = c-np.dot(affine,ocenter)
        return interpolation.affine_transform(image,affine,offset=offset)

def find_number(img):
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    digit = cv2.bitwise_not(img)

    top, bottom, left, right = [50]*4  # Adjust these values as needed
    digit_with_border = cv2.copyMakeBorder(digit, top, bottom, left, right, 
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    cv2.imshow('Inverted Digit', digit_with_border)
    cv2.waitKey(0)

    ocr_output = pytesseract.image_to_string(digit_with_border, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    print(ocr_output)
    text = ocr_output.replace("\n", "")

    if(text != ''):
        text = int(text)
    else:
        text = 0
    print("KLASA",  text)
    return text
    



app.run(host="127.0.0.1", port=8003, debug=True)