from model import EmotionCNNLSTM
import torch
import torch.nn as nn
from dataset import ESD
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device, epoch=0):
    model.train()
    running_loss = 0
    for mel_spec,emotion,speaker_id,language in tqdm(train_loader):
        mel_spec, emotion = mel_spec.to(device), emotion.to(device)

        optimizer.zero_grad()
        outputs = model(mel_spec)
        loss = criterion(outputs, emotion)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    return running_loss/len(train_loader)

def evaluate(model, test_loader, criterion, device):
    val_loss = 0
    correct= 0 
    total = 0
    model.eval()
    for mel_spec,emotion,speaker_id,language in tqdm(test_loader):
        mel_spec, emotion = mel_spec.to(device), emotion.to(device)
        outputs = model(mel_spec)
        loss = criterion(outputs, emotion)
        val_loss += loss.item()
        predicted = torch.argmax(outputs, 1)
        correct += (predicted == emotion).sum().item()
        total += emotion.size(0)
        
    accuracy = 100 * correct / total
    print(f"Val Loss: {val_loss/len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return val_loss/len(test_loader), accuracy



train_dataset = ESD("train_set.csv")
test_dataset = ESD("test_set.csv")

model = EmotionCNNLSTM(5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 256
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss()
df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "accuracy"])

epochs = 2001

for epoch in range(epochs):
    train_loss = train(model,trainloader,criterion,optimizer,device,epoch)
    val_loss,acc = None, None
    if epoch % 10 == 0:
        val_loss,acc = evaluate(model,testloader,criterion,device)
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f"./model/CNNLSTM{epoch}.pth")

    new_row = pd.DataFrame([[epoch, train_loss, val_loss, acc]], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("train_log.csv", index=False)