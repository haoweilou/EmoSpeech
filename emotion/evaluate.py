import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import collate_fn


from dataset import ESD
from model import EmotionCNNLSTM,EmotionCNN

def evaluate(model, test_loader, criterion, device):
    val_loss = 0
    correct = 0
    total = 0
    
    # Dictionary to store correct and total counts per (language, emotion) pair
    lang_emotion_correct = defaultdict(int)
    lang_emotion_total = defaultdict(int)
    
    # Dictionary to store correct and total counts per language
    language_correct = defaultdict(int)
    language_total = defaultdict(int)
    
    model.eval()
    
    for mel_spec, emotion, speaker_id, language in tqdm(test_loader):
        mel_spec, emotion = mel_spec.to(device), emotion.to(device)
        outputs = model(mel_spec)
        loss = criterion(outputs, emotion)
        val_loss += loss.item()
        predicted = torch.argmax(outputs, 1)
        
        # Overall accuracy tracking
        correct += (predicted == emotion).sum().item()
        total += emotion.size(0)
        
        # Accuracy per (language, emotion) pair and per language
        for lang, emo, pred, true in zip(language, emotion.cpu().numpy(), predicted.cpu().numpy(), emotion.cpu().numpy()):
            key = (lang, emo)
            lang_emotion_total[key] += 1
            if pred == true:
                lang_emotion_correct[key] += 1
            
            language_total[lang] += 1
            if pred == true:
                language_correct[lang] += 1
    
    # Compute overall accuracy
    accuracy = 100 * correct / total
    
    # Compute accuracy per (language, emotion) pair
    lang_emotion_acc = {key: 100 * lang_emotion_correct[key] / lang_emotion_total[key] for key in lang_emotion_total}
    
    # Compute accuracy per language
    language_acc = {lang: 100 * language_correct[lang] / language_total[lang] for lang in language_total}
    
    print(f"Val Loss: {val_loss/len(test_loader):.4f}, Overall Accuracy: {accuracy:.2f}%")
    print("Accuracy by (Language, Emotion) Pair:", lang_emotion_acc)
    print("Accuracy by Language:", language_acc)
    
    from dataset import emo_dict, lang_dict
    emo_dict = {emo_dict[k]: k for k in emo_dict}
    lang_dict = {lang_dict[k]: k for k in lang_dict}

    for keys in sorted(lang_emotion_acc.keys()):
        print(f"{lang_dict[keys[0]]} - {emo_dict[keys[1]]}: {lang_emotion_acc[keys]:.2f}")
    
    return val_loss/len(test_loader), accuracy, lang_emotion_acc, language_acc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_dataset = ESD("test_set.csv")

# model = EmotionCNNLSTM(5).to(device)
# model.load_state_dict(torch.load("./model/CNNLSTM1600.pth"))
model = EmotionCNN(5).to(device)
model.load_state_dict(torch.load("./model/CNN2000.pth"))
batch_size = 32
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss()

evaluate(model,testloader,criterion,device)