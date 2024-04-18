from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__ ,static_url_path='/static')
model = YOLO('best.pt')
####
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    result = model.predict(source="Thảm hoạ cháy nổ trong năm 2015 - VTC.mp4", imgsz=640, conf=0.5, show=True)
    return result.imgs[0]

if __name__ == '__main__':
    app.run(debug=True)
