import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd

# Class names
class_names = ['related', 'unrelated']

# Set the pretrained vocabulary tokenizer from Huggingface library.
VOCAB_NAME = 'bert-base-uncased'

# Set the pretrained BERTModel from Huggingface library.
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# Set DataLoader hyperparameters
MAX_LENGTH = 80
BATCH_SIZE_TRAINING = 16
BATCH_SIZE_VALIDATION = 256
BATCH_SIZE_TESTING = 256

# Set model hyperparameters
LEARNING_RATE=2e-5
EPOCHS = 3

# Model filename
saved_model_dict = './models/model_epochs30Sept/model_state_2.pth'

class SequenceClassifier(nn.Module):
    
    def __init__(self, n_classes, pretrained_model):
        super(SequenceClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = bert_output[1]
        output = self.drop(pooled_output)
        return self.out(output)

class ModelClassifier():
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB_NAME)
        
        classifier = SequenceClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
        classifier.load_state_dict(
            torch.load(saved_model_dict, map_location=self.device))
        
        classifier.eval()
        self.classifier = classifier.to(self.device)
        
    def predict(self, text_one, text_two):
        encoded_text = self.tokenizer(
            text_one,
            text_two,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].to(self.device)
        token_type_ids = encoded_text['token_type_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        
        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, token_type_ids, attention_mask), dim=1)
        
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": confidence,
            "probabilities": dict(zip(class_names, probabilities))
        }