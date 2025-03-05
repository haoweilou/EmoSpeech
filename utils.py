import torch
def collate_fn(batch):
    mel_spec,emotion,speaker_id,language = zip(*batch)

    # Get maximum time length in batch
    max_len = max([mel.shape[-1] for mel in mel_spec])  # b, n_mels, times
    # Zero-pad Mel spectrograms
    padded_melspecs = torch.zeros(len(batch), mel_spec[0].shape[1], max_len)  # (batch, n_mels, times)

    for i, mel in enumerate(mel_spec):
        padded_melspecs[i, : , :mel.shape[-1]] = mel
    
    speaker_id = torch.tensor(speaker_id, dtype=torch.long)
    emotion = torch.tensor(emotion, dtype=torch.long)
    padded_melspecs = torch.unsqueeze(padded_melspecs,1)
    return padded_melspecs, emotion,speaker_id,language