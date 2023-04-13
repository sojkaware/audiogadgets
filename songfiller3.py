import os
import librosa
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

Set up the Wav2Vec2 model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

Define the sliding window size and hop length
window_size = 5 # in seconds
hop_length = 2.5 # in seconds
gap_size = 0.1 # in seconds

Define the length of each label segment
label_length = 5 - gap_size # in seconds

Define the sample rate
sample_rate = 16000

Define the directory containing the songs
input_dir = 'songs'

Define the directories to store the training and testing data
train_dir = 'data_training'
test_dir = 'data_testing'

Iterate over each song in the input directory
for filename in os.listdir(input_dir):

# Load the song using Librosa
song_path = os.path.join(input_dir, filename)
y, sr = librosa.load(song_path, sr=sample_rate, mono=True)

# Compute the total number of frames in the song
num_frames = int(np.ceil(len(y) / (hop_length * sample_rate)))

# Iterate over each frame in the song
for frame_idx in range(num_frames):
    
    # Compute the start and end times for the current window
    start_time = frame_idx * hop_length
    end_time = start_time + window_size
    
    # Extract the audio for the current window
    window_audio = y[int(start_time * sample_rate):int(end_time * sample_rate)]
    
    # Iterate over each label segment in the window
    for label_idx in range(int(window_size / label_length)):
        
        # Compute the start and end times for the current label segment
        label_start_time = start_time + label_idx * label_length
        label_end_time = label_start_time + label_length
        
        # Extract the audio for the current label segment
        label_audio = window_audio[int(label_start_time * sample_rate):int(label_end_time * sample_rate)]

# Introduce a gap in the middle of the label audio
gap_start = int((label_length / 2 - gap_size / 2) * sample_rate)
gap_end = int((label_length / 2 + gap_size / 2) * sample_rate)
gap_audio = np.concatenate((label_audio[:gap_start], label_audio[gap_end:]))

# Save the gap audio and label audio as .wav files
output_dir = train_dir if np.random.rand() < 0.8 else test_dir
gap_filename = f"{filename[:-4]}_gap_{frame_idx}_{label_idx}.wav"
label_filename = f"{filename[:-4]}_label_{frame_idx}_{label_idx}.wav"
gap_filepath = os.path.join(output_dir, gap_filename)
label_filepath = os.path.join(output_dir, label_filename)

torchaudio.save(gap_filepath, torch.tensor(gap_audio).unsqueeze(0), sample_rate)
torchaudio.save(label_filepath, torch.tensor(label_audio).unsqueeze(0), sample_rate)
Fine-tune the Wav2Vec2 model
from datasets import load_dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

Define the processor, tokenizer, and feature extractor
processor = Wav2Vec2Processor(feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'),
tokenizer=Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h'))

Load the training and testing data
train_dataset = load_dataset('wav2vec2_data_preparation.py', 'data_training', split='train')
test_dataset = load_dataset('wav2vec2_data_preparation.py', 'data_testing', split='test')

Prepare the dataset for fine-tuning
def prepare_dataset(batch):
inputs = processor(batch["input_values"], sampling_rate=batch["sampling_rate"], return_tensors="pt", padding=True)
with processor.as_target_processor():
labels = processor(batch["labels"], return_tensors="pt", padding=True)
batch["input_values"] = inputs.input_values[0]
batch["labels"] = labels.input_ids
return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

Define the training arguments
training_args = TrainingArguments(
output_dir='./results',
num_train_epochs=3,
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
evaluation_strategy='epoch',
save_strategy='epoch',
logging_strategy='epoch',
logging_first_step=True,
learning_rate=1e-4,
fp16=True,
save_total_limit=2,
)

Instantiate the Trainer
trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset,
tokenizer=processor.feature_extractor,
)

Train the model
trainer.train()

Save the fine-tuned model
model.save_pretrained('fine_tuned_wav2vec2')
tokenizer.save_pretrained('fine_tuned_wav2vec2')