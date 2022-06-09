from flask import Flask, render_template, request, redirect, flash, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os
import cv2
from network.main import image_detect

app = Flask(__name__)
Bootstrap(app)

ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]
UPLOAD_FOLDER = "static/upload/"

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def is_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():  # put application's code here
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)

    if file and is_allowed(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        image = image_detect(path)
        cv2.imwrite(path, image)
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)

    # print(gunicorn.__version__)
    flash("Allowed image types are: png, jpg, jpeg")
    return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename=("upload/" + filename)), code=301)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
