import nltk
from nltk.corpus import cmudict
# Example usage
import re
import os 
# nltk.download('cmudict') # run on drstrange
nltk_data_dir = os.path.abspath("./") #run on gadi
nltk.data.path.append(nltk_data_dir)
alphabet_to_ipa = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}
# print(len(alphabet_to_ipa.keys()))
try:
    pronouncing_dict = cmudict.dict()
    print("CMU Pronouncing Dictionary loaded successfully!")
except LookupError:
    print("CMU Pronouncing Dictionary not found in the specified path.")
    print("Ensure the directory structure is correct: nltk_data/corpora/cmudict/")
    
ipa_to_word = {}

def word_to_phoneme(word):
    word_lower = word.lower()  # CMU Dict uses lowercase words
    if word_lower in pronouncing_dict:
        # Return the first pronunciation for simplicity
        return pronouncing_dict[word_lower][0]
    else:
        return None  # Word not found in the dictionary

def phoneme_to_ipa(phonemes):
    return [alphabet_to_ipa.get(p) for p in phonemes]

def pho_stress_split(phonemes):
    stress = []
    phoneme = []
    for pho in phonemes:
        last = pho[-1]
        if str.isnumeric(last): 
            stress.append(int(last))
            phoneme.append(pho[:-1])
        else: 
            stress.append(0)
            phoneme.append(pho)
    return phoneme,stress


def word_to_ipa(word):
    #vowel contain 0,1,2 to symbolize the stress
    #0, no stress, 1 primary stress, 2 secondary stress
    global ipa_to_word
    if word == "|": return ["|"],[0]
    elif word == '’':return ["’"],[0]

    phonemes = word_to_phoneme(word)
    phoneme, stress = pho_stress_split(phonemes)
    ipa_phoneme = phoneme_to_ipa(phoneme)
    if "".join(ipa_phoneme) not in ipa_to_word:
        ipa_to_word["".join(ipa_phoneme)] = word
    return ipa_phoneme, stress

def normalize_sentence(sentence):
    words = sentence.split(" ")
    output = []
    for word in words: 
        if word in pronouncing_dict:
            output.append([word])
        else: 
            output.append(split_into_words(word,pronouncing_dict))
    # output = " ".join(output)
    return output
    
def normalize_sentence_with_seg(sentence):
    words = sentence.split(" ")
    output = []
    for word in words: 
        if word in pronouncing_dict or word == "|":
            output.append([word])
        else: 
            output.append(split_into_words(word,pronouncing_dict))
    return output

def split_into_words(word, word_dictionary):
    word = word.lower()
    result = []
    i = 0
    while i < len(word):
        found = False
        for j in range(len(word), i, -1): 
            segment = word[i:j]
            if segment in word_dictionary:
                result.append(segment)
                i = j - 1
                found = True
                break
        if not found:  
            result.append(word[i])
        i += 1
    return result

def english_to_alphabet(sentence:str):
    sentence = sentence.lower()
    words = sentence.split(" ")

    alphabet_phonemes = ["|"]
    stresses = [0]
    for word in words:
        for char in word: 
            if not char.isalpha(): continue
            alphabet_phonemes.append(char)
            stresses.append(0)
        alphabet_phonemes.append("|")
        stresses.append(0)
    return alphabet_phonemes, stresses


def english_segmented_to_ipa(sentence):
    sentence = normalize_sentence_with_seg(sentence)
    ipa_phonemes = ["START"]
    stresses = [0]
    for word in sentence:
        stress = []
        word_phoneme = []
        for subword in word:
            ipa, s = word_to_ipa(subword)
            word_phoneme+=ipa
            stress += s

        ipa_phonemes += word_phoneme
        stresses += stress

    ipa_phonemes.append("END")
    stresses.append(0)
    return ipa_phonemes, stresses

def english_segmented_symbol_to_ipa(sentence):
    sentence = normalize_sentence_with_seg(sentence)
    ipa_phonemes = ["START"]
    stresses = [0]
    for word in sentence:
        stress = []
        word_phoneme = []
        for subword in word:
            if subword in symbols:
                ipa = [subword]
                s = [0]
            else: 
                ipa, s = word_to_ipa(subword)
            word_phoneme+=ipa
            stress += s

        ipa_phonemes += word_phoneme
        stresses += stress

    ipa_phonemes.append("END")
    stresses.append(0)
    return ipa_phonemes, stresses



def english_symbol_style_to_ipa(sentence):
    ipa_phonemes = ["START"]
    stresses = [0]
    forward_styles = [0]
    forward_scales = [0]
    backward_styles = [0]
    backward_scales = [0]

    for i,phrase in enumerate(sentence):
        stress = []
        word_phoneme = []

        forward_symbol = phrase[0]["f"]
        backward_symbol = phrase[0]["b"]
        forward_style_idx = ipa_style_symbol_dict[forward_symbol] if forward_symbol != 'N' else 0
        backward_style_idx = ipa_style_symbol_dict[backward_symbol] if backward_symbol != 'N' else 0

        # words = phrase[1]
        words = phrase[1].split()
        output = []
        for word in words: 
            if word in pronouncing_dict or word == "|":
                output.append(word)
            else: 
                output += split_into_words(word,pronouncing_dict)

        words = output
        for subword in words:
            ipa, s = word_to_ipa(subword)
            word_phoneme+=ipa
            stress += s

        ipa_phonemes += word_phoneme
        stresses += stress
        num_phoneme = len(word_phoneme)
        forward_styles += [forward_style_idx] * num_phoneme
        backward_styles += [backward_style_idx] * num_phoneme

        if backward_style_idx == 0:
            backward_scales += [0 for _ in range(num_phoneme)]
        else: 
            backward_scales += [round(0 + (scale+1)/num_phoneme,2) for scale in range(num_phoneme)]

        if forward_style_idx == 0:
            forward_scales += [0 for _ in range(num_phoneme)]
        else: 
            forward_scales += [round(1 - (scale)/num_phoneme,2) for scale in range(num_phoneme)]

        if i != len(sentence) - 1:
            ipa_phonemes.append("|")
            stresses.append(0)
            forward_styles.append(0)
            forward_scales.append(0)
            backward_styles.append(0)
            backward_scales.append(0)            

    ipa_phonemes.append("END")
    stresses.append(0)
    forward_styles.append(0)
    forward_scales.append(0)
    backward_styles.append(0)
    backward_scales.append(0)   
    assert len(ipa_phonemes) == len(stresses) and len(forward_styles) == len(backward_styles) and len(ipa_phonemes) == len(forward_styles)
    assert len(backward_styles) == len(forward_styles) and len(backward_styles) == len(ipa_phonemes)
    return ipa_phonemes, stresses, forward_styles, backward_styles, forward_scales, backward_scales

def english_sentence_to_ipa(sentence):
    # print(sentence)
    sentence = normalize_sentence(sentence)
    #[['the'], ['shameful'], ['malpractice', 's'], ['of'], ['bam', 'bridge']]
    #english
    ipa_phonemes = ["START"]
    stresses = [0]
    for word in sentence:
        # print(word)
        # ipa_phoneme = []
        stress = []
        word_phoneme = []
        for subword in word:
            ipa, s = word_to_ipa(subword)
            word_phoneme+=ipa
            stress+=s

        ipa_phonemes += word_phoneme
        ipa_phonemes.append("|")

        stresses += stress
        stresses.append(0)
    ipa_phonemes[-1] = "END"
    return ipa_phonemes, stresses

#phoneme, number of ipa phoneme
#stress, 3, 0: no stress, 1 for primary stress, 2 for non-stress, mostly applied to vowel
#tone

pinyin_to_ipa = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", 
    "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x",
    "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ɻ",
    "z": "ts", "c": "tsʰ", "s": "s",
    "w": "w", "y": "j",
    "a": "a", "o": "o", "e": "ɤ", "i": "i", "u": "u", "v": "y",
    "ai": "aɪ", "ei": "eɪ", "ao": "aʊ", "ou": "oʊ",
    "an": "an", "en": "ən", "ang": "ɑŋ", "eng": "əŋ", "ong": "ʊŋ",
    "er": "ɚ",
    "ia": "ja", "iao": "jaʊ", "ie": "je", "iu": "joʊ",
    "ian": "jan", "in": "in", "iang": "jɑŋ", "ing": "iŋ",
    "ua": "wa", "uo": "wo", "uai": "waɪ", "ui": "weɪ",
    "uan": "wan", "un": "wən", "uang": "wɑŋ", "ue": "yɛ", "van": "yan", "vn": "yn",
    "ve":"yɛ"
}

from pypinyin import pinyin, Style
def hanzi_to_pinyin(hanzi):
    return [syllable[0] for syllable in pinyin(hanzi, style=Style.TONE3)]
                                               
initials = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "w", "y"
]
finals = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    "er"
]

def pinyin_to_phoneme(syllable):
    initial = None
    if syllable[:2] in initials: 
        initial = syllable[:2]
        final = syllable[2:]
    elif syllable[:1] in initials:
        initial = syllable[:1]
        final = syllable[1:]
    else: 
        final = syllable
    tone = final[-1] if final[-1].isdigit() else "0"
    if final[-1].isdigit(): final = final[:-1]
    if final == "iong": 
        final = ["i","ong"]
    else: 
        final = [final]
    if "" in final: final.remove("")
    if len(final) == 0: 
        phoneme = [initial]
        tones = [int(tone)]
    else: 
        phoneme = [initial]+final if initial is not None else final
        tones = [0]*(len(phoneme)-1)+[int(tone)]
    return phoneme, tones

def pinyin_to_ipa_phoneme(syllable):
    phoneme, tones = pinyin_to_phoneme(syllable)
    ipa_phoneme = [pinyin_to_ipa.get(p) for p in phoneme]
    return ipa_phoneme, tones

def pinyin_sentence_to_ipa(pinyin_sentence):
    ipa_phonemes = ["START"]
    tones = [0]
    for pinyin in pinyin_sentence:
        ipa_phoneme, tone = pinyin_to_ipa_phoneme(pinyin)
        ipa_phoneme.append("|")
        tone.append(0)
        ipa_phonemes += ipa_phoneme
        tones += tone
    ipa_phonemes[-1] = "END"
    return ipa_phonemes, tones


def segment_pinyin_sentence_to_ipa(pinyin_sentence):
    ipa_phonemes = ["START"]
    tones = [0]
    for pinyin in pinyin_sentence:
        if len(pinyin) == 1:
            ipa_phonemes.append("|")
            tones.append(0)
        else: 
            ipa_phoneme, tone = pinyin_to_ipa_phoneme(pinyin)
            ipa_phonemes += ipa_phoneme
            tones += tone
    ipa_phonemes.append("END")
    tones.append(0)

    return ipa_phonemes, tones

def segment_symbol_pinyin_sentence_to_ipa(pinyin_sentence):
    ipa_phonemes = ["START"]
    tones = [0]
    for pinyin in pinyin_sentence:
        if pinyin == " ":
            ipa_phonemes.append("|")
            tones.append(0)
        elif not pinyin[0].isalpha():
            symbols = []
            tone = []
            for symbol in pinyin:
                symbols.append(symbol)
                tone.append(0)
            ipa_phonemes += symbols
            tones += tone
        else: 
            ipa_phoneme, tone = pinyin_to_ipa_phoneme(pinyin)
            ipa_phonemes += ipa_phoneme
            tones += tone
    ipa_phonemes.append("END")
    tones.append(0)

    return ipa_phonemes, tones


def chinese_symbol_style_to_ipa(ch_sentence):
    ipa_phonemes = ["START"]
    tones = [0]
    forward_styles = [0]
    forward_scales = [0]
    backward_styles = [0]
    backward_scales = [0]

    for i,phrase in enumerate(ch_sentence):
        ipa_phoneme = []
        tone = []

        forward_symbol = phrase[0]["f"]
        backward_symbol = phrase[0]["b"]
        forward_style_idx = ipa_style_symbol_dict[forward_symbol] if forward_symbol != 'N' else 0
        backward_style_idx = ipa_style_symbol_dict[backward_symbol] if backward_symbol != 'N' else 0

        ch_chars = phrase[1]
        pinyins = hanzi_to_pinyin(ch_chars.strip())
        for pinyin in pinyins:
            ipa, ton = pinyin_to_ipa_phoneme(pinyin)
            ipa_phoneme +=ipa
            tone += ton
            
        # print(ipa_phoneme,tone)

        ipa_phonemes += ipa_phoneme
        tones += tone
        num_phoneme = len(ipa_phoneme)
        forward_styles += [forward_style_idx] * num_phoneme
        backward_styles += [backward_style_idx] * num_phoneme

        if backward_style_idx == 0:
            backward_scales += [0 for _ in range(num_phoneme)]
        else: 
            backward_scales += [round(0 + (scale+1)/num_phoneme,2) for scale in range(num_phoneme)]

        if forward_style_idx == 0:
            forward_scales += [0 for _ in range(num_phoneme)]
        else: 
            forward_scales += [round(1 - (scale)/num_phoneme,2) for scale in range(num_phoneme)]

        if i != len(ch_sentence) - 1:
            ipa_phonemes.append("|")
            tones.append(0)
            forward_styles.append(0)
            forward_scales.append(0)
            backward_styles.append(0)
            backward_scales.append(0)            

    ipa_phonemes.append("END")
    tones.append(0)
    forward_styles.append(0)
    forward_scales.append(0)
    backward_styles.append(0)
    backward_scales.append(0)   
    assert len(ipa_phonemes) == len(tones) and len(forward_styles) == len(backward_styles) and len(ipa_phonemes) == len(forward_styles)
    assert len(backward_styles) == len(forward_styles) and len(backward_styles) == len(ipa_phonemes)
    return ipa_phonemes, tones, forward_styles, backward_styles, forward_scales, backward_scales


    
def mandarin_chinese_to_ipa(sentence):
    pinyin_sentence = hanzi_to_pinyin(sentence)
    return pinyin_sentence_to_ipa(pinyin_sentence)

def mandarin_segment_to_ipa(sentence):
    pinyin_sentence = hanzi_to_pinyin(sentence)
    return segment_pinyin_sentence_to_ipa(pinyin_sentence)

def mandarin_segmented_symbol_to_ipa(sentence):
    pinyin_sentence = hanzi_to_pinyin(sentence)
    ipa_pho,style = segment_symbol_pinyin_sentence_to_ipa(pinyin_sentence)
    if None in ipa_pho: print(sentence,pinyin_sentence,ipa_pho,sep="\n")
    return ipa_pho,style


def mandarin_list_to_phrase_segmented_str(sentence:list):
    output = ""
    for phrase in sentence: 
        if phrase == " ":
            output += " | "
        elif phrase in style_symbols: 
            output += " " + phrase+" "
        else: 
            output += phrase
    return output

def mandarin_chinese_to_alpha(sentence):
    pinyins = hanzi_to_pinyin(sentence)

    alphabet_phonemes = ["|"]
    tones = [0]
    for pinyin in pinyins:
        for char in pinyin: 
            if not char.isalpha(): continue
            alphabet_phonemes.append(char)
            tones.append(0)
        alphabet_phonemes.append("|")
        tones.append(0)
    return alphabet_phonemes, tones

def ipa_to_idx(ipa_phonemes):
    return  [ipa_pho_dict[i] for i in ipa_phonemes]   

def remove_symbols(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)  # Keep only A-Z, a-z, 0-9, and spaces

all_ipa_phoneme = list(pinyin_to_ipa.values())+list(alphabet_to_ipa.values())
all_ipa_phoneme = sorted(list(set(all_ipa_phoneme)))
all_ipa_phoneme = ["EMPTY"] + all_ipa_phoneme+["|","START","END"]
ipa_pho_dict = {k:i for i,k  in enumerate(all_ipa_phoneme)}
idx_to_ipa = {i:k for i,k  in enumerate(all_ipa_phoneme)}

import string
alpha_list = ["EMPTY"] + [a for a in string.ascii_lowercase] + ["|"]
alpha_pho_dict = {k:i for i,k in enumerate(alpha_list)}
# print("IPA Phoneme Dict:",ipa_pho_dict)
# print("Alpha Phoneme Dict:",alpha_pho_dict)
import jieba
def segment_chinese(sentence):
    """Quickly segment Chinese phrases using Jieba (efficient rule-based method)."""
    return " ".join(jieba.cut(sentence))

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
def segment_english(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)

    segmented = []
    current_phrase = []
    
    for word, tag in tagged:
        current_phrase.append(word)
        if tag in ['CC', 'IN', 'DT', '.', ',']:  # Split at conjunctions, prepositions, and punctuation
            segmented.append(" ".join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        segmented.append(" ".join(current_phrase))
    
    return " | ".join(segmented)  # Use "|" to show phrase breaks


symbol_normalizer = {
    "？":"?","'":"’","。":".",'（':"(", '）':")", '，':",",'：':":", '；':";","—":"-","！":"!","`":"'",'"':""
}

symbols = ['!', '"', "’", '(', ')', ',', '-', '.', ':', ';', '?', '[', ']', '“', '”', '…', '、']
all_ipa_phoneme_symbols = all_ipa_phoneme+symbols
ipa_symbol_dict = {k:i for i,k  in enumerate(all_ipa_phoneme_symbols)}

phoneme_symbols = ["’"]
style_symbols = ['neutral','!', ',', '-', '.', ':', ';', '?',  '…', '、']
all_ipa_phoneme_symbols_v2 = all_ipa_phoneme+phoneme_symbols

ipa_symbol_dict_v2 = {k:i for i,k  in enumerate(all_ipa_phoneme_symbols_v2)}
ipa_style_symbol_dict = {k:i for i,k  in enumerate(style_symbols)}

all_alpha_phoneme = [a for a in string.ascii_lowercase] + [" "]
alpha_dict = {k:i for i,k in enumerate(all_alpha_phoneme)}