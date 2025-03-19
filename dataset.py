import torch
import pandas as pd 
import os
import torchaudio
import torchaudio.transforms as transforms

def load_wav_to_torch(filename):
    """Loads a .wav file to a PyTorch tensor."""
    waveform, sampling_rate = torchaudio.load(filename)
    return waveform, sampling_rate

emo_dict = {"neutral":0,"angry":1,"happy":2,"sad":3,"surprise":4}
lang_dict = {"ch":0,"en":1}
class ESD(torch.utils.data.Dataset):
    def __init__(self, csv_path,sample_rate=16000, n_mels=80,sep=","):
        super(ESD, self).__init__()
        self.data = pd.read_csv(csv_path,sep=sep)
        # self.data["file_path"] = self.data["file_path"].apply(lambda x: x.replace("C:/","D:/"))
        self.data["file_path"] = self.data["file_path"].apply(lambda x: x.replace("C:/","/share/scratch/haoweilou/"))
        self.sample_rate = sample_rate

        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256
        )


    def __getitem__(self, index):
        row = self.data.iloc[index]
        file_path = row["file_path"]
        speaker_id = row["speaker"]
        emotion = row["emotion"]
        emotion = emo_dict[emotion]

        text = row["text"]
        language = "ch" if any(ord(c) > 127 for c in text) else "en"
        language = 0 if language == "ch" else 1

        mel_spec = self._load_mel_spectrogram(file_path)
        return mel_spec,emotion,speaker_id,language

    def _get_spec_path(self, audio_path):
        """Generates the spectrogram filename from the original WAV filename."""
        return audio_path.replace(".wav", ".spec.pt")

    def _load_mel_spectrogram(self, audio_path):
        """Loads audio, resamples if necessary, normalizes, and retrieves Mel spectrogram."""
        spec_path = self._get_spec_path(audio_path)

        if os.path.exists(spec_path):
            return torch.load(spec_path)
        else: 
            audio, sr = load_wav_to_torch(audio_path)
            audio_norm = audio.unsqueeze(0)  # Add channel dim
            spec = self.mel_transform(audio_norm)
            spec = torch.squeeze(spec, 0)

            # Save for future use
            torch.save(spec, spec_path)
            return spec

    def __len__(self):
        return len(self.data)