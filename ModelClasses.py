from transformers import BertModel, BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import re

import time
from datetime import timedelta

import config

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
        
        self.tokenizer = BertTokenizer.from_pretrained(config.VOCAB_NAME)
        
        classifier = SequenceClassifier(len(config.class_names), config.PRE_TRAINED_MODEL_NAME)
        classifier.load_state_dict(
            torch.load(config.MODEL_DIR, map_location=self.device))
        
        classifier.eval()
        self.classifier = classifier.to(self.device)
    
    def _preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
        text = re.sub(r'<[^<]+?>', '', text)
        text = text.lower()
        return text
    
    def predict(self, text_one, text_two):
        text_one = self._preprocess_text(text_one)
        text_two = self._preprocess_text(text_two)
        
        encoded_text = self.tokenizer(
            text_one,
            text_two,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
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
            "predicted_class": config.class_names[predicted_class],
            "confidence": confidence,
            "probabilities": dict(zip(config.class_names, probabilities))
        }
    
    def model_evaluation(self, data_loader):
        y_true = []
        y_pred = []
        accuracy_per_batch = []
        
        i_counter = 1
        
        evaluation_time_start = time.time()
        
        with torch.no_grad():
            for d in data_loader:
                batch_time_start = time.time()
                input_ids = d['input_ids'].to(self.device)
                token_type_ids = d['token_type_ids'].to(self.device)
                attention_mask = d['attention_mask'].to(self.device)
                label = d['label'].to(self.device)
                
                outputs = self.classifier(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                )
                
                _, predicted_class = torch.max(outputs, dim=1)
                
                y_true = y_true + label.tolist()
                y_pred = y_pred + predicted_class.tolist()
                accuracy_per_batch.append(torch.sum(predicted_class == label)/config.BATCH_SIZE_FINAL_EVALUATION)
                
                batch_time_elapsed = time.time() - batch_time_start
                print(f'Batch {i_counter}/{len(data_loader)}')
                print(f'Accuracy {(torch.sum(predicted_class == label))/config.BATCH_SIZE_FINAL_EVALUATION}')
                print(f'Batch runtime {batch_time_elapsed // 60:.0f}m {batch_time_elapsed % 60:.0f}s')
                
                i_counter += 1
        
        evaluation_time_elapsed = time.time() - evaluation_time_start
        print(f'Evaluation runtime {evaluation_time_elapsed // 60:.0f}m {evaluation_time_elapsed % 60:.0f}s')
        
        return y_true, y_pred, accuracy_per_batch