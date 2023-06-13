import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import azure.ai.vision as sdk
from PIL import Image
import sys
import time
from io import BytesIO
import cv2
import gradio as gr
from PIL import Image
import base64
from gpt4all import GPT4All
from moviepy.editor import *
import threading
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import streamlit as st
import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
import cv2
from PIL import Image
import time
import openai
from keys import OPENAI_API_KEY, azure_vision_key
openai.api_key = OPENAI_API_KEY

class video_reader():
    def __init__(self,video_path) -> None:
        self.cap = cv2.VideoCapture(video_path)

    def extract_frame(self, timestamp):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        target_frame = int(fps * timestamp)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception(f"Failed to read frame {target_frame}")
        cv2.imwrite("a.jpg", frame)
        return f"a.jpg"

class musicgen_meta():
    def __init__(self) -> None:
        pass
        self.model = MusicGen.get_pretrained('melody')
        self.model.set_generation_params(duration=10)
        self.count = 0
        
    def generate_audio(self, descriptions):
        if self.count == 1:
            self.melody, self.sr = torchaudio.load('./0.wav')
            wav = self.model.generate_with_chroma(descriptions, self.melody[None].expand(1, -1, -1), self.sr)
        else:
            wav = self.model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(f'{idx}', one_wav.cpu(), self.model.sample_rate, strategy="loudness")
        count = 1
        return f'0.wav'


def play_audio_with_fade():
    audio = AudioSegment.from_wav("0.wav")
    audio_fade_out = audio.fade_out(200)
    audio_fade_in = audio.fade_in(200)
    combined_audio = audio_fade_out.append(audio_fade_in)

    play(combined_audio)
service_options = sdk.VisionServiceOptions("https://computer-vision-vaze.cognitiveservices.azure.com/",
                                           azure_vision_key)

class azure_image_analysis():
    def __init__(self) -> None:
        pass
    def get_captions(self,image):
        vision_source = sdk.VisionSource(filename="a.jpg")
        # vision_source = sdk.VisionSource(
        #     url="https://learn.microsoft.com/azure/cognitive-services/computer-vision/media/quickstarts/presentation.png")
        analysis_options = sdk.ImageAnalysisOptions()
        analysis_options.features = (
            sdk.ImageAnalysisFeature.CAPTION
        )
        analysis_options.language = "en"

        analysis_options.gender_neutral_caption = True

        image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

        result = image_analyzer.analyze()
        if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:

            if result.caption is not None:
                print("Caption:"+result.caption.content)
                return result.caption.content
        
        return "no caption"

def openai_api(query):
    try:
        system_query =  "Reply with a prompt for an AI based music generation for the image description which I send. Use these categories to describe the music in a short paragraph: instruments, moods, sounds, genres, rhythms, harmonies, melodies, tempo, emotion. Only reply the prompt in 2 lines."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #model="gpt-4",
            messages=[{"role": "system", "content": system_query},
                      {"role": "user", "content": "image description: " + query}]
        )
        #print(response)
        res = response["choices"][0]["message"]["content"].replace('\'', '')
        
        return res
    except:
        return "no gpt response"

def local_ai_api(query):
    gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")
    # messages = [{"role": "user", "content": "Name 3 colors"}]
    # gptj.chat_completion(messages)
    system_query =  "Reply with a prompt for an AI based music generation for the image description which I send. Use these categories to describe the music in a short paragraph: instruments, moods, sounds, genres, rhythms, harmonies, melodies, tempo, emotion. Only reply the prompt in 2 lines."
    response = gptj.chat_completion(
            messages=[{"role": "user", "content": system_query + ".image description: " + query}]
        ,
        streaming=False)
        #print(response)
    res = response["choices"][0]["message"]["content"].replace('\'', '')
    return res

def run():
    vdeo_file = "b.mp4"
    music_start_time = time.time()
    m = musicgen_meta()
    a = azure_image_analysis()
    c = video_reader(vdeo_file)
    start_time = time.time()
    captions = openai_api(a.get_captions(c.extract_frame(5.0)))
    print(captions)
    p = m.generate_audio(["A melody loop for" + captions])
    response_time = time.time() - start_time
    print(f"Request took {response_time:.2f} seconds")
    audio = AudioSegment.from_wav("0.wav")
    video = VideoFileClip(vdeo_file)
    # Load audio file
    audio = AudioFileClip("0.wav")
    # Set the video's audio
    video_with_new_soundtrack = video.set_audio(audio)
    # Save the result
    video_with_new_soundtrack.write_videofile("output.mp4")

def main():
    local_ai_api("apple")
if __name__ == "__main__":
    run()