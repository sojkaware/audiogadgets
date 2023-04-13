
# Here is a Python script that transcribes audio files using the facebook/wav2vec2-base model and detects questions in the transcriptions:
# pip install transformers librosa spotipy beautifulsoup4    
# pip install podcast-downloader

# secrets
import credentials

import os
import librosa
import sounddevice as sd
import torch
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Spot
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
#import spotipy.util as util
import requests
from bs4 import BeautifulSoup


#import podcast_downloader
#import spotdl

#from podcast_downloader import downloaded

import os


# for listennotes
import json
import re
from listennotes import podcast_api
import requests


# this works
def download_podcast_listennotes(p_name, folder_dl):
    # Initialize the ListenNotes API client
    client = podcast_api.Client(api_key=credentials.LISTENNOTES_KEY)

    response = client.search(q=p_name, type='episode')
    if response.json()['results'] == []:
        print(f"No podcast episode found for: '{p_name}'.")
        return

    # # Open file in write mode and write JSON string to file
    # with open("response.json", "w") as file:
    #     file.write(json.dumps(response.json()))
    #     file.close() 

    # Get the first result and extract the audio URL
    episode = response.json().get("results", [])[0]
    audio_url = episode["audio"]

    # Extract the episode title and use it as the filename
    title = re.sub(r'\s+', '_', episode['title_original'])
    filename = f"{title}.mp3"

    # Create the download folder if it doesn't exist
    if not os.path.exists(folder_dl):
        os.makedirs(folder_dl)

    # Download the audio file
    filepath = os.path.join(folder_dl, filename)
    headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
    audio_file = requests.get(audio_url, headers=headers)

    with open(filepath, "wb") as f:
        f.write(audio_file.content)

    print(f"Downloaded '{filename}' to '{folder_dl}'.")



###########################Spotipy does not work yet
# https://towardsdatascience.com/extracting-song-data-from-the-spotify-api-using-python-b1e79388d50
# import spotipy.oauth2 as oauth2
# import credentials

# SCOPE = 'user-library-read'
# CACHE = '.spotipyoauthcache'

# sp_oauth = oauth2.SpotifyOAuth(
#     credentials.SPOTIPY_CLIENT_ID, 
#     credentials.SPOTIPY_CLIENT_SECRET, 
#     credentials.SPOTIPY_REDIRECT_URI, 
#     scope=SCOPE, 
#     show_dialog=True, 
#     cache_path=CACHE
# def download_podcast_spotipy(spotify_url):
#     # Set up Spotipy credentials
#     client_id = 'your_client_id'
#     client_secret = 'your_client_secret'
#     # username = 'your_username'
#     # scope = 'user-library-read'
#     # token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri='http://localhost/')

#     sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="YOUR_APP_CLIENT_ID",
#                                                             client_secret="YOUR_APP_CLIENT_SECRET"))

#     results = sp.search(q='weezer', limit=20)
#     for idx, track in enumerate(results['tracks']['items']):
#         print(idx, track['name'])


#     # Initialize Spotipy object
#     sp = spotipy.Spotify(auth=token)

#     # Get the podcast ID from the URL
#     res = requests.get(spotify_url)
#     soup = BeautifulSoup(res.content, 'html.parser')
#     meta = soup.find_all('meta')
#     for m in meta:
#         if m.get('property') == 'twitter:player:stream':
#             url = m.get('content')
#             podcast_id = url.split('/')[-1]
#             break

#     # Get the audio file URL
#     audio_file = sp._get('https://api.spotify.com/v1/episodes/{}'.format(podcast_id))['audio_preview_url']

#     # Download the audio file
#     r = requests.get(audio_file)
#     with open('podcast.mp3', 'wb') as f:
#         f.write(r.content)
#     return 'Podcast downloaded successfully'


## This is spotdl method. It can download some episodes only
# url = 'https://open.spotify.com/episode/0RPGop8CzYhOWRAAxNFxVW?si=2ca55a3c897b44a3'
# # Some address not work. Only podcasts that are on youtube maybe?
# #url = 'https://open.spotify.com/track/254bXAqt3zP6P50BdQvEsq?si=e9a5afba89b347cc'
# folder = 'C:\\Users\\Emsušenka\\gitsojkaware\\audiogadgets\\output_txt'

# # Never use the argument: --ffmpeg "C:/ffmpeg/bin"   otherwise you get PermissionError: [WinError 5] Access is denied    
# # Let spotdl install it
# os.system(f'spotdl {url} --output "{folder}" --format "mp3" --bitrate "128k"  --dont-filter-results')




# this works with easier inputs
def transcribe_audio(audio, sr, model_name="facebook/wav2vec2-large-960h-lv60"):
    # Load the pre-trained Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Resample the audio to the expected sampling rate
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)

    # Preprocess the audio
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    # Perform the transcription
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription




# def transcribe_audio(model, tokenizer, audio):
#     inputs = tokenizer(audio, return_tensors="pt", padding="longest").input_values
#     with torch.no_grad():
#         logits = model(inputs).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = tokenizer.batch_decode(predicted_ids)[0]
#     return transcription

# def detect_questions(transcription):
#     sentences = transcription.split(".")
#     is_questions = []

#     for sentence in sentences:
#         is_question = sentence.strip().endswith("?")
#         is_questions.append(is_question)

#     return is_questions

folder_podcasts_mp3 = "downloaded_podcasts_mp3"
folder_text = "podcasts_txt"
folders = [folder_text, folder_podcasts_mp3]
for folder in [folder_text, folder_podcasts_mp3]:
    if not os.path.exists(folder):
        os.makedirs(folder)

#episode_podcast_name = "Avi Goldfarb The Economic Impact of AI Invest"
#download_podcast_listennotes(episode_podcast_name, "downloaded_podcasts_mp3")
#print('Downloaded.')

#model_name = "facebook/wav2vec2-base"
#tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
#tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
#model = Wav2Vec2ForCTC.from_pretrained(model_name)

for filename in os.listdir(folder_podcasts_mp3):
    audio_path = os.path.join(folder_podcasts_mp3, filename)
    audio, sr = librosa.load(audio_path, sr=16000)
    #sd.play(audio, sr)

    # Transcribe the audio
    transcribed_text = transcribe_audio(audio, sr)

    # # Save the transcription as a .txt file
    # with open("output_transcription.txt", "w") as outfile:
    #     outfile.write(transcribed_text)
    # transcribed_text = transcribe_audio(model, tokenizer, audio)

   # is_questions = detect_questions(transcribed_text)

    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_txt_filepath = os.path.join(folder_text, output_filename)

    with open(output_txt_filepath, "w") as f:
        f.write(transcribed_text)

    print(f"Transcription saved to: {output_txt_filepath}")
    #print(f"Is questions: {is_questions}")



