# audio.py

from flask import route, request, Response
import queue
import sounddevice as sd
from pydub.audio_segment import AudioSegment


import random
# importing custom python module
import base64
from io import BytesIO
from matplotlib.figure import Figure
import librosa

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    return render_template('index.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    spectrogram_data = request.data
    data = [time()*1000, random.random()*100]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response


if __name__ == "__main__":
    input_stream(1)
