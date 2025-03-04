import os
import pandas as pd
path = "C:/Emotion Speech Dataset/"

speakers = os.listdir(path)
ch_speakers = [i for i in range(1,11)]
en_speakers = [i for i in range(11,21)]

def ch_emo_to_en(ch:str):
    if ch == "中立":
        return "neutral"
    elif ch == "生气":
        return "angry"
    elif ch == "快乐":
        return "happy"
    elif ch == "伤心":
        return "sad"
    elif ch == "惊喜":
        return "surprise"

all_data = []

for ch_speaker in ch_speakers:
    speaker_dir = f'{path}{ch_speaker:04d}/'
    log_path = f"{speaker_dir}{ch_speaker:04d}.txt"
    data = pd.read_csv(log_path,sep='\t',header=None)
    data.columns = ['file_name',"text","emotion"]
    data["emotion"] = data["emotion"].apply(ch_emo_to_en)
    data["speaker"] = ch_speaker
    data["file_path"] = f"{speaker_dir}" + data["emotion"].apply(lambda x:x.capitalize()) + "/" + data['file_name'] + ".wav"
    for item in data["file_path"]:
        assert os.path.exists(item)
    all_data.append(data)

for en_speaker in en_speakers:
    speaker_dir = f'{path}{en_speaker:04d}/'
    log_path = f"{speaker_dir}{en_speaker:04d}.txt"
    data = pd.read_csv(log_path,sep='\t',header=None)
    data.columns = ['file_name',"text","emotion"]
    data["emotion"] = data["emotion"].apply(lambda x:x.lower().strip())
    data["speaker"] = en_speaker
    data["file_path"] = f"{speaker_dir}" + data["emotion"].apply(lambda x:x.capitalize()) + "/" + data['file_name'] + ".wav"
    for item in data["file_path"]:
        assert os.path.exists(item)
    all_data.append(data)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("esd.csv",sep="\t",index=None)