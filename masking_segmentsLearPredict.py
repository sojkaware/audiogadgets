#pip install torchaudio transformers

import os
import random
import numpy as np
import torchaudio
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Processor
import torch
from torch.optim import Adam

# Parameters
input_songs_folder = "input_songs_mp3"
input_songs_prediction_folder = "input_songs_prediction_mp3"
sample_rate = 16000  # 16kHz
segment_length = 3 * sample_rate  # 3 seconds
mask_length = int(0.1 * sample_rate)  # 100 ms
split_ratio = 0.9

# Load the model and processor from Hugging Face
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_path, sample_rate):
    waveform, _ = torchaudio.load(file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)(waveform)
    return waveform

# Load all input audio files
input_audio_files = [
    os.path.join(input_songs_folder, f) for f in os.listdir(input_songs_folder) if f.endswith(".mp3")
]

# Load and preprocess audio files
waveforms = [load_and_preprocess_audio(file_path, sample_rate) for file_path in input_audio_files]

# Split audio files into segments and mask the middle of each segment
segments = []
for waveform in waveforms:
    num_segments = len(waveform[0]) // segment_length
    for i in range(num_segments):
        segment = waveform[:, i * segment_length:(i + 1) * segment_length]
        masked_segment = segment.clone()
        masked_segment[:, (segment_length // 2) - (mask_length // 2):(segment_length // 2) + (mask_length // 2)] = 0
        segments.append((masked_segment, segment))

# Split data into training and testing/validation sets
random.shuffle(segments)
split_idx = int(len(segments) * split_ratio)
train_segments = segments[:split_idx]
test_segments = segments[split_idx:]

# Prepare the model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
optimizer = Adam(model.parameters())

# Training loop
for masked_segment, segment in train_segments:
    # Convert audio to features
    input_values = processor(masked_segment, return_tensors="pt").input_values.to(device)
    labels = processor(segment, return_tensors="pt").input_values.to(device)

    # Forward pass and compute loss
    outputs = model(input_values, labels=labels)
    loss = outputs.loss

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation on test set
model.eval()
with torch.no_grad():
    total_loss = 0
    for masked_segment, segment in test_segments:
        input_values = processor(masked_segment, return_tensors="pt").input_values.to(device)
        labels = processor(segment, return_tensors="pt").input_values.to(device)

        outputs = model(input_values, labels=labels)
        total_loss += outputs.loss.item()

    average_loss = total_loss / len(test_segments)
    print(f"Test loss: {average_loss}")

# Predict masked segments in unseen data
prediction_audio_files = [
    os.path.join(input_songs_prediction_folder, f) for f in os.listdir(input_songs_prediction_folder) if f.endswith(".mp3")
]

# Load and preprocess audio files
prediction_waveforms = [load_and_preprocess_audio(file_path, sample_rate) for file_path in prediction_audio_files]

# Process each prediction waveform
for waveform in prediction_waveforms:
    num_segments = len(waveform[0]) // segment_length
    for i in range(num_segments):
        segment = waveform[:, i * segment_length:(i + 1) * segment_length]
        masked_segment = segment.clone()
        masked_segment[
            :, (segment_length // 2) - (mask_length // 2):(segment_length // 2) + (mask_length // 2)
        ] = 0

        # Perform prediction
        with torch.no_grad():
            input_values = processor(masked_segment, return_tensors="pt").input_values.to(device)
            prediction = model(input_values).logits

        # Post-process prediction and reconstruct waveform
        predicted_waveform = processor.inverse(prediction)
        reconstructed_segment = torch.cat(
            (segment[:, :(segment_length // 2) - (mask_length // 2)], predicted_waveform[0],
            segment[:, (segment_length // 2) + (mask_length // 2):]),
            dim=1,
        )

        # Combine reconstructed segments into final waveform
        if i == 0:
            reconstructed_waveform = reconstructed_segment
        else:
            reconstructed_waveform = torch.cat((reconstructed_waveform, reconstructed_segment), dim=1)

    # Save reconstructed waveform to file
    file_name = os.path.basename(os.path.normpath(prediction_audio_files[0])).split(".")[0] + ".wav"
    output_path = os.path.join("output_songs", file_name)
    torchaudio.save(output_path, reconstructed_waveform, sample_rate)
