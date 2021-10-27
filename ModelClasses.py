"""
Contains all classes related to the model.

SequenceClassifer class was used for model training.

ModelClassifier class is the final model.
"""

# Import BertModel and BertTokenizer
from transformers import BertModel, BertTokenizer

# Import torch modules
import torch
from torch import nn
import torch.nn.functional as F

# Import regex
import re

# Import time
import time

# Import config
import config

class SequenceClassifier(nn.Module):
    """
    Main object for the BERT model.
    Extended from the pytorch nn module. Dictates the forward pass.
    
    Parameters
    ==========
    n_classes: Integer
        The number of classes.
    
    pretrained_model: String
        The name of the BertModel
        
    Attributes
    ==========
    bert: BertModel
        The BERT model object.
    
    drop: DropOut
        The DropOut object from pytorch nn
    
    out: Linear
        The Linear object from pytorch nn
    """
    
    def __init__(self, n_classes, pretrained_model):
        super(SequenceClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Forward pass.
        
        Parameters
        ==========
        input_ids: list
            The input ids tokens.
        
        token_type_ids: list
            The token type tokens.
        
        attention_mask: list
            The attention mask tokens.
            
        Returns
        =======
        
        out: Linear
            The Linear object.
        """
        
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = bert_output[1]
        output = self.drop(pooled_output)
        return self.out(output)

class ModelClassifier():
    """
    Final model object.
    This object should hold the final object.
    Ensure the model directory is correct in config.
    
    Attributes
    ==========
    device: torch.device
        GPU if available, otherwise it will perform with CPU.
    
    tokenizer: BertTokenizer
        The tokenizer.
    
    classfier: BertModel
        The final model.
    """
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained(config.VOCAB_NAME)
        
        classifier = SequenceClassifier(len(config.class_names), config.PRE_TRAINED_MODEL_NAME)
        classifier.load_state_dict(
            torch.load(config.MODEL_DIR, map_location=self.device))
        
        classifier.eval()
        self.classifier = classifier.to(self.device)
    
    def _preprocess_text(self, text):
        """
        Text preprocessing.
        Remove every special character and HTML elements.
        Lowercase all string.
        
        Parameters
        ==========
        text: String
            Raw string from user input.
            
        Returns
        =======
        
        text: String
            Preprocessed string.
        """
        text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
        text = re.sub(r'<[^<]+?>', '', text)
        text = text.lower()
        return text
    
    def predict(self, text_one, text_two):
        """
        Predict the relation between two texts.
        
        Steps:
        1) Preprocess the texts.
        2) Tokenize the texts.
        3) Predict the text with loaded model.
        
        Parameters
        ==========
        
        text_one: String
            Raw string from user input as the first sentence.
        
        text_two: String
            Raw string from user input as the second sentence.
            
        Returns
        =======
        
        output: Dictionary
            A dictionary of the evaluation results.
            
            keys:
                predicted_class: String
                    The resultant prediction.
                confidence: float
                    The confidence of the model to its answer.
                probabilities: list
                    Final values of the output layer.
        """
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
    
    def print_prediction(self, text_one, text_two):
        """
        Prints out the results from the predict method.
        
        Parameters
        ==========
        
        text_one: String
            Raw string from user input as the first sentence.
        
        text_two: String
            Raw string from user input as the second sentence.
            
        Returns
        =======
        
        output: String
            The prediction as a String
        """
        results = self.predict(text_one, text_two)
        
        results_txt = f'Text One:  {text_one}'
        results_txt = results_txt + f'\nText Two:  {text_two}'
        results_txt = results_txt + '\n'
        results_txt = results_txt + f'\nPrediction: {results["predicted_class"]}'
        results_txt = results_txt + f'\nConfidence: {results["confidence"].item()}'
        results_txt = results_txt + '\nPrediction Probabilities'
        for key in results["probabilities"]:
            results_txt = results_txt + f'\n{key}: {results["probabilities"][key]}'
        
        return results_txt
    
    def model_evaluation(self, data_loader):
        """
        Evaluate a pytorch DataLoader object.
        
        Parameters
        ==========
        
        data_loader: DataLoader
            Pytorch DataLoader of a testing set.
            
        Returns
        =======
        
        y_true: list
            A list of the correct label.
        
        y_pred: list
            A list of the predicted label.
        
        accuracy_per_batch: list
            A list of the accuracy per batch.
        """
        
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