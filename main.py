from flask import Flask, Response, make_response, send_from_directory
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename

from someMethods import parseTrain, Tasks
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('mainui.html')

@app.route('/favicon.ico',methods=['GET'])
def getIco():
    return make_response(send_from_directory("templates","sharefolder_desktop_link.ico",as_attachment=True))

@app.route('/test/upload',methods=['POST'])
def processUpload():
    save_dir = request.headers.get("Path")
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    uploaded_files  = request.files.getlist("file")
    [file.save(os.path.join(save_dir,secure_filename(file.filename))) for file in uploaded_files]
    return 'ok'


@app.route('/newTask',methods=['POST'])
def addNewTask():
    try:
        cmd = parseTrain(request.form.to_dict())
        return t.addTask(cmd)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    t = Tasks()
    app.run(port=7860)
