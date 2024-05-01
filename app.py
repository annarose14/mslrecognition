from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import math
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageFont, ImageDraw

app = Flask(__name__)
loaded_model = load_model("model_msl.keras")
detector = HandDetector()
let_list = ["അ", 'ആ', 'എ', 'ഇ', 'ക', 'ണ', 'ല', 'ള', 'മ', 'ന', 'ഒ', 'പ', 'ര', 'റ', 'ഋ', 'ത', 'ഉ', 'വ']
offset = 20
imz = 256
sent = ""
previous_r = ""
count = 0

font_path = 'AnjaliOldLipi-Regular.ttf'  # Specify the path to your Malayalam TTF font file
font = ImageFont.truetype(font_path, 32)

def gen_frames():
    cap = cv2.VideoCapture(0)
    global previous_r, sent  # Declare previous_r and sent as global
    while True:
        success, img = cap.read()
        hands, img_d = detector.findHands(img)
        imgBac = np.ones((imz, imz, 3), np.uint8) * 255
        gray = np.ones((imz, imz, 1), np.uint8) * 255
        try:
            for hand in hands:
                if hand['type'] == 'Right':
                    x, y, w, h = hand['bbox']
                    img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    ratio = h / w
                    if ratio > 1:
                        k = imz / h
                        wCal = math.ceil(w * k)
                        imgr = cv2.resize(img_crop, (wCal, imz))
                        wgap = math.ceil((imz - wCal) / 2)
                        imgBac[:, wgap:wCal + wgap] = imgr
                    else:
                        k = imz / w
                        hCal = math.ceil(h * k)
                        imgr = cv2.resize(img_crop, (imz, hCal))
                        hgap = math.ceil((imz - hCal) / 2)
                        imgBac[hgap:hCal + hgap, :] = imgr
                    res = cv2.resize(imgBac, (128, 128))
                    result = loaded_model.predict(res.reshape(1, 128, 128, 3), verbose=0)
                    out = np.argmax(result)
                    pred = let_list[out]
                    if pred == previous_r:
                        count += 1
                    else:
                        count = 0
                    if count > 10:
                        count = 0
                        sent += pred
                    previous_r = pred

                    # Draw bounding box
                    cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 2)

                    # Display recognized letters on the image
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x, y - 50), sent, font=font, fill=(0,0,0))
                    img = np.array(img_pil)

        except Exception as e:
            print(f"Error: {e}")

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def favicon():
    return Response("", status=200)

if __name__ == '__main__':
    app.run(debug=True)
