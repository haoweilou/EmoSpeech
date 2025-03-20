from tts.model import EmoSpeech
import torch
from ipa import all_ipa_phoneme
from utils import model_size, get_hparams
hps = get_hparams()
n_style = 8
n_emotion = 5
model = EmoSpeech(len(all_ipa_phoneme),
    n_style,
    n_emotion,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
size = model_size(model)
print(size)
# print(model)
phone = torch.randint(0,len(all_ipa_phoneme),(2,30)).cuda()
phone_len = torch.tensor([30,20]).cuda().long()
style = torch.randint(0,n_style,(1,30)).cuda()
emotion = torch.tensor([[1,0,0,0,0],[1,0.5,0,0.5,0]]).cuda().long()
spectrogram = torch.randn(2,513,50).cuda()
spectrogram_len = torch.tensor([50,45]).cuda()
speaker_id = torch.tensor([0,1]).cuda().long()
output = model(phone,style,phone_len,spectrogram,spectrogram_len,speaker_id,emotion)