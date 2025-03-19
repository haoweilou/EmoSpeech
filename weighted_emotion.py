from dataset import ESD
import pandas as pd 
from model import EmotionCNNLSTM
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate_fn
import torch
import torch.nn as nn
import  torch.nn.functional as F

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNNLSTM(5).to(device)
model.load_state_dict(torch.load("./model/CNNLSTM2000.pth"))
emo_dict = {"neutral":[],"angry":[],"happy":[],"sad":[],"surprise":[]}
data_df = pd.read_csv("./esd.csv",sep="\t")

dataset = ESD("esd.csv",sep="\t")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
for mel_spec,emotion,speaker_id,language in tqdm(dataloader):
    mel_spec, emotion = mel_spec.to(device), emotion.to(device)
    outputs = model(mel_spec)
    outputs = F.softmax(outputs,dim=1)
    for output in outputs:
        emotion_score = [round(value.item(), 2) for value in output.cpu().detach()]
        emo_dict["neutral"].append(emotion_score[0])
        emo_dict["angry"].append(emotion_score[1])
        emo_dict["happy"].append(emotion_score[2])
        emo_dict["sad"].append(emotion_score[3])
        emo_dict["surprise"].append(emotion_score[4])


emo_df = pd.DataFrame(emo_dict)
data_df = pd.concat([data_df, emo_df], axis=1)
data_df.to_csv("./esd_weighted_emotion.csv",index=False,sep='\t')
