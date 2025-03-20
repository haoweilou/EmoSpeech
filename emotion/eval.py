import pandas as pd 
import numpy as np
from dataset import emo_dict
data = pd.read_csv('esd_weighted_emotion.csv',sep='\t')
data['emotion'] = data['emotion'].map(emo_dict)
data["predict_emotion"] = data.apply(lambda x: np.argmax([x["neutral"],x['angry'],x['happy'],x['sad'],x['surprise']]), axis=1)
data_filtered = data[['emotion','predict_emotion']]
accuracy = (data_filtered["emotion"] == data_filtered["predict_emotion"]).mean()

print(f"accuracy: {accuracy*100:.2f}")