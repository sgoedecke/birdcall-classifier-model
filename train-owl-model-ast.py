# too hard, too annoying for now
# https://www.kaggle.com/code/benmcewen1/possum-detector
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score 
from datasets import load_dataset, Audio, concatenate_datasets

# Run on GPU (cuda:0) if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current running on {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
classifier = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

save_path = "pretrained_ast_path"
# !mkdir {save_path}

feature_extractor.save_pretrained(save_path)
classifier.save_pretrained(save_path)

class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, target_rate=16000, val=False):
        super(AudioDataset, self).__init__()
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        self.audio_dir = audio_dir
        self.target_rate = target_rate
        self.resize = transforms.Resize((224,224))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        resample = torchaudio.transforms.Resample(sr, self.target_rate)
        signal = resample(signal[0])
        feature = self.feature_extractor(signal, sampling_rate=self.target_rate, return_tensors="pt")
        feature = feature['input_values']
        return feature,label
    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,1])
        return path
    def _get_audio_label(self, index):
        return self.annotations.iloc[index,3]
    def _get_validation(self, index):
        return self.annotations.iloc[index,2]
    def _filter_annotations(self, data):
        return data.loc[data['Validation'] == self.val]
    def shuffle(self):
        pass

class Model(nn.Module):
    # Create user model for transfer learning
    def __init__(self):
        super(Model, self).__init__()
        in_features = 527
        self.linear1 = torch.nn.Linear(in_features, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def encode(labels, classes):
    target = []
    for label in labels:
        y = [0,0]
        y[classes.index(label)] = 1
        target.append(y)
    target = torch.Tensor(target)
    return target

def train(training_set, validation_set, classes, num_epoch=1, batch_size=2):
    model = Model()
    model = model.to(device)
    print('Generating model') #
    ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") #
    ast = ast.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = [] 
    best_acc = 0
    learning_rate = 1e-6
    # Recommended hyper-parameters - epoch:25, lr:1e-5 (halving every 5 epochs after epoch 10), batch:12
    for epoch in range(num_epoch):
        optimizer = optim.Adam(list(ast.parameters()) + list(model.parameters()), lr=learning_rate)
        if epoch > 2:
            learning_rate = learning_rate/2
        running_loss = 0
        running_corrects = 0
        val_running_loss = 0
        corrects = 0
        total = 0
        i = 0
        GT = []
        pred = []
        for row in training_set:
            data = torch.tensor(row['input_values'])
            label = row['label']
            data = data.to(device)
            data = torch.squeeze(data,1)
            optimizer.zero_grad()
            embeddings = ast(data).logits
            outputs = model(embeddings)
            y = encode([label], classes) # list wrapper as label not batched
            y = y.to(device)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total += len(torch.argmax(y,dim=1))
            corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            running_corrects = 100*corrects/total
            accuracy = running_corrects.item()
            running_loss += loss.item()
        train_loss.append(running_loss)
        train_accuracy.append(accuracy)
        val_running = 0
        val_total = 0
        val_corrects = 0
        i = 0
        for row in validation_set:
            data = torch.tensor(row['input_values'])
            label = row['label']
            data = data.to(device)
            data = torch.squeeze(data,1)
            embeddings = ast(data).logits
            outputs = model(embeddings)
            y = encode([label], classes)
            y = y.to(device)
            loss = criterion(outputs, y)
            val_total += len(torch.argmax(y,dim=1))
            val_corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            val_running = 100*val_corrects/val_total
            val_running_accuracy = val_running.item()
            val_running_loss += loss.item()
            GT.append(list(y[0].cpu()).index(1))
            pred.append(torch.argmax(outputs,dim=1).cpu().item())
        val_loss.append(val_running_loss)
        val_accuracy.append(val_running_accuracy)  
    print('AST')
    print(f'[{epoch + 1}], Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/len(training_set):.2f}, Val Accuracy: {val_running_accuracy:.2f}, Val loss: {val_running_loss/len(validation_set):.2f}')
    print('F1: {}'.format(f1_score(pred, GT, average='macro')))
    print('Precision: {}'.format(precision_score(pred, GT, average='macro')))
    print('Recall: {}'.format(recall_score(pred, GT, average='macro')))
    print('--------------------------------------')
    return train_loss, train_accuracy, val_loss, val_accuracy

    # AUDIO_DIR = "/kaggle/input/invasive-species/training/training/"
    # ANNOTATIONS = "/kaggle/input/invasive-species/training/training/training.csv"

def preprocess_function(examples):
    target_length = 5 * feature_extractor.sampling_rate  # 5 seconds * sampling rate
    padded_audio_arrays = []
    for audio_array in examples["audio"]:
        current_length = len(audio_array["array"])
        if current_length < target_length:
            # Calculate the number of samples to pad
            padding_length = target_length - current_length
            # Pad with zeros (silence)
            padded_audio = np.pad(audio_array["array"], (0, padding_length), mode='constant')
        elif current_length > target_length:
            # Optionally truncate the clip to the target length
            padded_audio = audio_array["array"][:target_length]
        else:
            # No padding needed
            padded_audio = audio_array["array"]
        padded_audio_arrays.append(padded_audio)
    # Use the feature_extractor as before, now with padded_audio_arrays
    inputs = feature_extractor(padded_audio_arrays, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    return inputs

classes = ['owl', 'not_owl']
birdcalls_raw = load_dataset("sgoedecke/powerful_owl_5s_16k", cache_dir="~/sean-birds-testing-usw3")
birdcalls = birdcalls_raw.cast_column("audio", Audio(sampling_rate=16_000))
birdcalls['train'] = birdcalls['train'].map(preprocess_function, remove_columns="audio", batched=True)
birdcalls = birdcalls['train'].remove_columns(['filename']).train_test_split(test_size=0.3)
label2id = { "owl": 0, "not_owl": 1 }
id2label = { 0: "owl", 1: "not_owl" }
print('Generating training/validation sets...')
training = birdcalls['train']#AudioDataset(ANNOTATIONS, AUDIO_DIR, target_rate=16000, val=False)
validation = birdcalls['test']#AudioDataset(ANNOTATIONS, AUDIO_DIR, target_rate=16000, val=True)
print('training...')
loss, acc, val_loss, val_acc = train(training, validation, classes)
    
