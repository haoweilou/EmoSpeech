import pandas as pd
import re
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import unicodedata

import jieba


from ipa import phoneme_symbols, style_symbols,ipa_symbol_dict_v2,ipa_style_symbol_dict, english_symbol_style_to_ipa,chinese_symbol_style_to_ipa,mandarin_list_to_phrase_segmented_str
print(ipa_symbol_dict_v2)
print(ipa_style_symbol_dict)

kept_symbols = phoneme_symbols + style_symbols
#this one no segmentation


from ipa import symbol_normalizer,english_to_ipa

def lower(sentence:str):
    sentence = sentence.lower()
    return en_norm(sentence)

def en_norm(sentence:str):
    sentence = "".join([i if i not in symbol_normalizer  else symbol_normalizer[i] for i in sentence])
    return sentence.replace("'", "’")

def remove_chars(text):
    return re.sub(r'[a-z\u4e00-\u9fff]', '', text)

def remove_symbols(text):
    pattern = f"[^{re.escape(''.join(kept_symbols))}\w\s]"
    #remove all symbols that is not part of keeped_symbols
    return re.sub(pattern, '', text)

def ch_normalize(text):
    return text.replace("--","-")

def check_language(text):
    if re.search(r'[a-z]', text):
        return 'en'
    elif re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    else:
        return 'other'
    
if __name__ == '__main__':
    # ljspeech = pd.read_csv('E:/LJSpeech\metadata.csv',delimiter=r'\s*\|\s*',header=None,engine="python",encoding="utf-8")
    # ljspeech.columns = ["name","s1","s2"]
    
    # ljspeech = ljspeech.drop(columns=["s1"])
    # ljspeech["sentence"] = ljspeech['s2'].apply(lower).apply(en_norm)
  
    # ljspeech["speaker"] = 0
    # ljspeech["file_path"] = "/scratch/ey69/hl6114/LJSpeech/wavs/" + ljspeech['name'] + ".wav"
    # ljspeech[['ipa','style']] = ljspeech["sentence"].apply(lambda x: pd.Series(english_to_ipa(x)))
    # ljspeech[['neutral','angry','happy','sad','surprise']] = [1,0,0,0,0] #assume all neutral
    # ljspeech['language'] = 'en'
    # ljspeech = ljspeech[['file_path','sentence','language',"speaker","ipa","style",'neutral','angry','happy','sad','surprise']]
    # ljspeech.to_csv("./fileloader/ljspeech.csv",index=None,sep='\t')
   
    esd = pd.read_csv("./emotion/esd_weighted_emotion.csv",sep='\t')
    esd['language'] = esd['text'].apply(check_language)
    esd['sentence'] = esd['text']
    esd = esd[['file_path','sentence','language',"speaker",'neutral','angry','happy','sad','surprise']]
    esd.to_csv("./fileloader/esd.csv",index=None,sep='\t')
    print(esd.head())
    # with open(f"L:/baker/ProsodyLabeling/000001-010000.txt","r",encoding="utf-8") as f: 
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
    # baker["speaker"] = 1
    # baker["sentence"] = baker["sentence"].apply(remove_symbols).apply(ch_normalize)
    # baker["sentence"] = baker["sentence"].apply(mandarin_list_to_phrase_segmented_str)
    # baker["symbol_style"] = baker['sentence'].apply(apply_symbol_style_to_phrase)

    # baker[['ipa_phoneme', 'precise_style','forward_style','backward_style',"forward_scales","backward_scales"]] = baker['symbol_style'].apply(lambda x: pd.Series(chinese_symbol_style_to_ipa(x)))

    # baker["forward_style"] = baker["forward_style"].apply(lambda x: [i+8 for i in x])
    # baker["backward_style"] = baker["backward_style"].apply(lambda x: [i+8 for i in x])

    # chars_and_symbols = set(''.join(ljspeech["sentence"])).union(set(''.join(baker["sentence"])))
    # symbols = re.sub(r'[a-z\u4e00-\u9fff]', '', "".join(chars_and_symbols))
    # symbols = sorted(symbols)
    # symbols = sorted(set(symbols) - set(['à', 'â', 'è', 'é', 'ê', 'ü']))
    # print(sorted(symbols))
    # ljspeech_selected = ljspeech[["file_path", "speaker", "sentence", "ipa_phoneme", "precise_style",'forward_style','backward_style',"forward_scales","backward_scales"]]
    # baker_selected = baker[["file_path", "speaker", "sentence", "ipa_phoneme", "precise_style",'forward_style','backward_style',"forward_scales","backward_scales"]]

    # output = pd.concat([ljspeech_selected, baker_selected], ignore_index=True)

    # output.to_csv("filelists/all_symbols_style.csv",index=None,sep='\t',header=None,)
    # ljspeech_train = ljspeech_selected.iloc[:9000]
    # ljspeech_test = ljspeech_selected.iloc[9000:10000]
    
    # baker_train  = baker_selected.iloc[:9000]
    # baker_test = baker_selected.iloc[9000:10000]

    # train_data = pd.concat([ljspeech_train, baker_train], ignore_index=True)
    # test_data = pd.concat([ljspeech_test, baker_test], ignore_index=True)

    # train_data.to_csv("filelists/train.csv",index=None,sep='\t',header=None,)
    # test_data.to_csv("filelists/test.csv",index=None,sep='\t',header=None,)