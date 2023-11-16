# Modules
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import sys
import os
import json
import wave
import configparser

SetLogLevel(0)

# Define Paths
current_directory_path = os.path.dirname(os.path.abspath(__file__))
vosk_language_model_path = os.path.join(current_directory_path, "..", "vosk_language_models")
ffmpeg_directory_path = os.path.join(current_directory_path, "ffmpeg")
configuration_path = os.path.join(current_directory_path, "configuration", "configuration.ini")

# Read Configuration File
def configurate():
    global current_directory_path, configuration_path

    reader = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    reader.read(configuration_path)

    return {
        'vosk_language_model': reader.get('GENERAL', 'vosk_language_model'),
        'input_directory': os.path.join(current_directory_path, reader.get('GENERAL', 'input_directory')),
        'output_directory': os.path.join(current_directory_path, reader.get('GENERAL', 'output_directory'))
    }

configuration_dictionary = configurate()

language_model = configuration_dictionary["vosk_language_model"]
input = configuration_dictionary["input_directory"]
output = configuration_dictionary["output_directory"]

# Add ffmpeg to the Environment
os.environ["PATH"] += os.pathsep + os.path.join(ffmpeg_directory_path)

# Convert audio files to .wav and mono
def audio_converter(input_file_path, output_dir_path):
    file_name, file_extension = os.path.splitext(os.path.basename(input_file_path))
    
    if file_extension.lower() not in [".wav", ".mp3"]:
        return None

    output_file_name = os.path.join(output_dir_path, f"{file_name}_mono.wav")

    if os.path.exists(output_file_name):
        print("Warning: File already exists")
    
    sound = AudioSegment.from_file(input_file_path)
    sound = sound.set_channels(1)
    sound.export(output_file_name, format = "wav")

    with wave.open(output_file_name, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Warning: Audio file must be WAV format mono PCM")
            return None

    return output_file_name

# Audio Files
def get_audio_files():
    global input

    file_list = []

    for file in os.listdir(input):
        if file.lower().endswith((".wav", ".mp3")):
            file_list.append(os.path.join(input, file))

    return file_list

audio_files = get_audio_files()

# Process Audio Files
def process_audio_files():
    global audio_files, output

    ans = []

    for file in audio_files:
        processed_file = audio_converter(file, output)
        if processed_file:
            ans.append(processed_file)

    if not ans:
        print("Error: No valid audio files found")
        sys.exit(1)

    return ans

processed_audio_files = process_audio_files()

# Vosk Model
vosk_model = Model(model_path = os.path.join(vosk_language_model_path, language_model))

# Results
def transcribe_results(file, model, output):
    with wave.open(file, "rb") as wf, \
        open(os.path.join(output, f"{os.path.splitext(os.path.basename(file))[0]}.txt"), 'w', encoding="utf-8") as txt_file, \
        open(os.path.join(output, f"{os.path.splitext(os.path.basename(file))[0]}.json"), 'w', encoding="utf-8") as json_file:

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)
        json_file.write("[")

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                jres = json.loads(rec.Result())
                txt_file.write(jres["text"] + ' ')
                json.dump(jres, json_file, ensure_ascii=False)
                json_file.write(",\n")

        jres = json.loads(rec.FinalResult())
        if jres["text"]:
            json.dump(jres, json_file, ensure_ascii=False)
            txt_file.write(jres["text"] + '\n')
        else:
            json_file.seek(json_file.tell() - 3, 0)
            json_file.truncate()

        json_file.write("\n]")

# Perform Transcription
for file_audio in processed_audio_files:
    transcribe_results(file_audio, vosk_model, output)