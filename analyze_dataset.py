import pandas as pd 
import librosa
from tqdm import tqdm
#Sample rate: 16000Hz

dataset = pd.read_csv("esd.csv",sep="\t")
sample_rate = None
durations = []
for file in tqdm(dataset["file_path"]):
    if sample_rate is None:
        y, sample_rate = librosa.load(file, sr=None)  # Load the file without resampling
        print(f"Sample Rate: {sample_rate}")
    duration = librosa.get_duration(filename=file)
    durations.append(duration)

dataset["duration"] = durations

# Determine language based on speaker ID
dataset["language"] = dataset["speaker"].apply(lambda x: "ch" if x <= 10 else "en")

# Summarize total disk size per language
size_summary = dataset.groupby("language")["duration"].sum().reset_index()
size_summary.columns = ["language", "duration"]
size_summary["duration"] = size_summary["duration"] / 3600

print(dataset)
print(size_summary)
# 2 langauge, 10 speaker each language, each speak 350 sentence,  with 5 emotion style
# Total = 2*10*350*5 = 35,000 samples, approx 29h of audio
#   language   duration
# 0       ch  15.658173
# 1       en  13.413030