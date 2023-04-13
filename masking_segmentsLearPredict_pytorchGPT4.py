import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchaudio

# Define parameters
sample_rate = 16000
segment_length = 3  # seconds
mask_length = 0.1  # seconds
input_dir = "input_songs_mp3"
train_ratio = 0.9

# Load and process audio data
def process_audio(input_dir):
    data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            song_path = os.path.join(input_dir, filename)
            y, sr = librosa.load(song_path, sr=sample_rate, mono=True)
            num_segments = int(np.ceil(len(y) / (sample_rate * segment_length)))
            
            for idx in range(num_segments):
                start = idx * segment_length * sample_rate
                end = (idx + 1) * segment_length * sample_rate
                segment = y[start:end]
                
                if len(segment) == segment_length * sample_rate:
                    data.append(segment)
                    
    return np.array(data)

audio_data = process_audio(input_dir)
np.random.shuffle(audio_data)
train_size = int(train_ratio * len(audio_data))
train_data, test_data = audio_data[:train_size], audio_data[train_size:]

# Create a custom dataset
class AudioDataset(Dataset):
    def __init__(self, data, mask_length, sample_rate):
        self.data = data
        self.mask_length = mask_length
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio = self.data[idx]
        mask_start = (len(audio) // 2) - (int(self.mask_length * self.sample_rate) // 2)
        mask_end = mask_start + int(self.mask_length * self.sample_rate)
        masked_audio = np.concatenate((audio[:mask_start], audio[mask_end:]))
        
        return torch.tensor(masked_audio, dtype=torch.float32), torch.tensor(audio, dtype=torch.float32)

# Create DataLoader
train_dataset = AudioDataset(train_data, mask_length, sample_rate)
test_dataset = AudioDataset(test_data, mask_length, sample_rate)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model
class ReconstructionModel(nn.Module):
    def __init__(self):
        super(ReconstructionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 1, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return x.squeeze(1)

model = ReconstructionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for masked_audio, audio in train_loader:
        optimizer.zero_grad()
        output = model(masked_audio)
        loss = criterion(output, audio)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "reconstruction_model.pth")