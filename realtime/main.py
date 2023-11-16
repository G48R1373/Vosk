# Modules
from vosk import Model, KaldiRecognizer
from datetime import datetime
import sounddevice as sd
import argparse
import queue
import sys
import json
import os
import requests
import threading
import configparser

# Init
CONFIG_FILE = 'configuration\\configuration.ini'
LOG_FILE = 'log_result.txt'
MODEL_DIR = "vosk_language_models"
server_ok = True
q = queue.Queue()

DEVICE_ID = 1  
SAMPLERATE = 16000 
RECORDING_FILENAME = None 

# Utility functions
def print_header(message, char='#', length=60):
    print(char * length)
    print(message)
    print(char * length)

def create_or_check_file(file_path):
    if not os.path.exists(file_path):
        open(file_path, "w", encoding="utf-8").close()

def send_result(msg, url, mode):
    global server_ok
    try:
        data = {'mode': mode, 'msg': msg}
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Server return error message: {response.status_code}")
            server_ok = False
    except Exception as e:
        print(f"Please check the connection toward the server --> {url}\nError: {e}")
        server_ok = True

# Main functions
def setup_configuration():
    conf_parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    conf_parser.read(os.path.join(script_path, CONFIG_FILE))

    config = {
        "server_url": conf_parser.get('GENERAL', 'server_url'),
        "mode": conf_parser.get('GENERAL', 'mode'),
        "stream_results": conf_parser.getboolean('GENERAL', 'stream_results'),
        "vosk_language_model": conf_parser.get('GENERAL', 'vosk_language_model'),
        "model_path": os.path.join(script_path, "..", MODEL_DIR)
    }
    return config

def setup_audio_stream(model_path, vosk_model, stream_results, server_url, mode):
    print_header("Recording Started")
    model = Model(model_path=os.path.join(model_path, vosk_model))
    dump_fn = open(RECORDING_FILENAME, "wb") if RECORDING_FILENAME else None

    with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=8000, device=DEVICE_ID,
                           dtype="int16", channels=1, callback=audio_callback):
        print_header("Press Ctrl+C to stop the recording")
        recognizer = KaldiRecognizer(model, SAMPLERATE)
        process_audio_stream(recognizer, dump_fn, stream_results, server_url, mode)

def audio_callback(indata, a, b, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def process_audio_stream(recognizer, dump_fn, stream_results, server_url, mode):
    while server_ok:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            msg = result.get("text", "")
            if msg:
                log_message(msg)
                if stream_results:
                    threading.Thread(target=send_result, args=[msg, server_url, mode]).start()
        if dump_fn:
            dump_fn.write(data)

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.writelines([timestamp, " ", message, '\n'])

def main():
    config = setup_configuration()

    create_or_check_file(LOG_FILE)
    setup_audio_stream(config['model_path'], config['vosk_language_model'], 
                       config['stream_results'], config['server_url'], config['mode'])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRecording Stopped")
    except Exception as e:
        sys.exit(f"{type(e).__name__}: {e}")
