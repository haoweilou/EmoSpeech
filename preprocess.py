import pandas as pd
import re
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
#this one no segmentation

from ipa import symbol_normalizer,english_to_ipa,chinese_to_ipa

def lower(sentence:str):
    sentence = sentence.lower()
    return en_norm(sentence)

def en_norm(sentence:str):
    sentence = "".join([i if i not in symbol_normalizer  else symbol_normalizer[i] for i in sentence])
    return sentence.replace("'", "’")

def ch_normalize(text):
    return text.replace("--","-")

def check_language(text):
    if re.search(r'[a-z]', text):
        return 'en'
    elif re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    else:
        return 'other'

def mix_to_ipa(text):
    if '£¬' in text: text = text.replace('£¬',',')
    if check_language(text) == 'en':
        return english_to_ipa(text)
    else:
        return chinese_to_ipa(text)



if __name__ == '__main__':
    target_column = ['file_path','sentence','language',"speaker","ipa","style",'neutral','angry','happy','sad','surprise']
    # ljspeech = pd.read_csv('E:/LJSpeech\metadata.csv',delimiter=r'\s*\|\s*',header=None,engine="python",encoding="utf-8")
    # ljspeech.columns = ["name","s1","s2"]
    
    # ljspeech = ljspeech.drop(columns=["s1"])
    # ljspeech["sentence"] = ljspeech['s2'].apply(lower).apply(en_norm)
  
    # ljspeech["speaker"] = 0
    # ljspeech["file_path"] = "/scratch/ey69/hl6114/LJSpeech/wavs/" + ljspeech['name'] + ".wav"
    # ljspeech[['ipa','style']] = ljspeech["sentence"].apply(lambda x: pd.Series(english_to_ipa(x)))
    # ljspeech[['neutral','angry','happy','sad','surprise']] = [1,0,0,0,0] #assume all neutral
    # ljspeech['language'] = 'en'
    # ljspeech = ljspeech[target_column]
    # ljspeech.to_csv("./fileloader/ljspeech.csv",index=None,sep='\t')
   
    esd = pd.read_csv("./emotion/esd_weighted_emotion.csv",sep='\t')
    esd['language'] = esd['text'].apply(check_language)
    esd['file_path'] = esd['file_path'].apply(lambda x: x.replace("C:/Emotion Speech Dataset/","/scratch/ey69/hl6114/ESD/"))
    esd['sentence'] = esd['text'].apply(lambda x: ''.join([symbol_normalizer[i] if i in symbol_normalizer  else i  for i in x]))
    esd[['ipa','style']] = esd['sentence'].apply(lambda x: pd.Series(mix_to_ipa(x)))
    esd = esd[target_column]
    esd.to_csv("./fileloader/esd.csv",index=None,sep='\t')
    print(esd.head())
    # with open(f"E:/baker/ProsodyLabeling/000001-010000.txt","r",encoding="utf-8") as f: 
    #     lines = f.readlines()
    #     hanzis = []
    #     for i in range(10000):
    #         sentence = re.sub(r'\d+|\#\d+', '', lines[i*2]).strip()
    #         sentence = "".join([i if i not in symbol_normalizer  else symbol_normalizer[i] for i in sentence])
    #         if "Ｂ" in sentence: sentence = sentence.replace("Ｂ","币")
    #         elif "Ｐ" in sentence: sentence = sentence.replace("Ｐ","批")
    #         hanzis.append(sentence)
    # baker = pd.DataFrame(hanzis, columns=['sentence'])
    # baker["file_path"] = [f"{i:06d}.wav" for i in range(1,10001)]
    # baker["file_path"] = "/scratch/ey69/hl6114/baker/Wave/" + baker["file_path"]
    # baker["speaker"] = 0
    # baker["sentence"] = baker["sentence"].apply(ch_normalize)
    # baker['language'] = 'zh'
    # baker[['neutral','angry','happy','sad','surprise']] = [1,0,0,0,0] #assume all neutral
    # baker[['ipa','style']] = baker['sentence'].apply(lambda x: pd.Series(chinese_to_ipa(x)))
    # baker = baker[target_column]
    # baker.to_csv("./fileloader/baker.csv",index=None,sep='\t')