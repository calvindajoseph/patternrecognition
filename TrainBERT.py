import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import numpy as np

from collections import defaultdict
import time
from datetime import timedelta

from FileManager import FileManager
from DatasetClasses import DataLoaderProcessor
from ModelClasses import SequenceClassifier

# Create FileManager instance.
fileManager = FileManager()

# Create DataLoaderProcessor instance.
data_loader = DataLoaderProcessor()

# Class names
class_names = ['related', 'unrelated']

# Load dataframe.
df = fileManager.load_preprocessed_fakenews_dataset_smaller(file_number=2)

# Set random state.
RANDOM_STATE = 42

# Set the pretrained vocabulary tokenizer from Huggingface library.
VOCAB_NAME = 'bert-base-uncased'

# Set the pretrained BERTModel from Huggingface library.
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# Set DataLoader hyperparameters
MAX_LENGTH = 100
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 128
BATCH_SIZE_TESTING = 128

# Set model hyperparameters
LEARNING_RATE=5e-5
EPOCHS = 3

# If GPU is available, set device to GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# If tokenizer available in models/tokenizer, set to True.
# Note: If tokenizer_downloaded set to False, the tokenizer will be downloaded.
# Note: If tokenizer_downloaded set to True, the tokenizer must exist within the library
tokenizer_downloaded = True
tokenizer = None

# Load tokenizer.
if not tokenizer_downloaded:
    print('Downloading Tokenizer. Please wait.')
    tokenizer = BertTokenizer.from_pretrained(VOCAB_NAME)
    print('Download Tokenizer complete.')
    tokenizer.save_pretrained('./models/tokenizer/')
else:
    tokenizer = BertTokenizer.from_pretrained('./models/tokenizer/')

# Separate features from target.
X = df.drop(['label'], axis=1)
y = df.label.to_numpy()

# Train, validation, test splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test)

# Explore the shapes of the feature shapes.
print(f'df_train shape: {X_train.shape}')
print(f'df_val shape: {X_val.shape}')
print(f'df_test shape: {X_test.shape}')

# Load data to torch DataLoader
train_data_loader = data_loader.fakenews_dataset_loader(X_train, y_train, tokenizer,MAX_LENGTH, BATCH_SIZE_TRAINING)
val_data_loader = data_loader.fakenews_dataset_loader(X_val, y_val, tokenizer,MAX_LENGTH, BATCH_SIZE_VALIDATION)
test_data_loader = data_loader.fakenews_dataset_loader(X_test, y_test, tokenizer,MAX_LENGTH, BATCH_SIZE_TESTING)

# Explore the first batch of the train_data_loader
data = next(iter(train_data_loader))
print(data.keys())
print(data['input_ids'].shape)
print(data['token_type_ids'].shape)
print(data['attention_mask'].shape)

# Establish a model
model = SequenceClassifier(n_classes=len(class_names), pretrained_model=PRE_TRAINED_MODEL_NAME)
# Load model to device (If no GPU, then CPU)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

# Train epoch
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, n_epochs):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    i_counter = 1
    
    for d in data_loader:
        batch_time_start = time.time()
        
        input_ids = d['input_ids'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
    
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
    
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (i_counter % 20) == 0:
            batch_time_elapsed = time.time() - batch_time_start
            print(f'Batch {i_counter}/{len(data_loader)}, Epoch {n_epochs + 1}/{EPOCHS}')
            print(f'Accuracy {(torch.sum(preds == labels))/BATCH_SIZE_TRAINING}, Loss {loss.item()}')
            print(f'Batch runtime {batch_time_elapsed // 60:.0f}m {batch_time_elapsed % 60:.0f}s')
        i_counter += 1
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

training_time_start = time.time()

for epoch in range(EPOCHS):
    
    epoch_time_start = time.time()
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(model, train_data_loader,
                                        loss_fn, optimizer, device,
                                        scheduler, len(X_train), epoch)
    print(f'Train loss: {train_loss}; Accuracy: {train_acc}')
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn,
                                   device, len(X_val))
    
    print(f'Val loss: {val_loss}; Accuracy: {val_acc}')
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    filename = f'models/model_epochs/model_state_{epoch}.pth'
    torch.save(model.state_dict(), filename)
    
    epoch_time_elapsed = time.time() - epoch_time_start
    print(f'Epoch runtime {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

train_time_elapsed = time.time() - training_time_start
train_time_delta = timedelta(seconds=int(train_time_elapsed))
print(f'Training runtime: {train_time_delta}')

testing_time_start = time.time()
test_acc, test_loss = eval_model(model, test_data_loader, loss_fn,
                                 device, len(X_val))

print(f'Test loss: {test_loss}; Accuracy: {test_acc}')
test_time_elapsed = time.time() - testing_time_start
test_time_delta = timedelta(seconds=int(test_time_elapsed))
print(f'Testing runtime: {test_time_delta}')

best_val_acc = 0
best_epoch = 0
for idx in range(EPOCHS):
    print(f'\nEpoch {idx + 1}')
    print('=======\n')
    print(f'Train loss: {train_losses[idx]}')
    print(f'Train acc: {train_accuracies[idx]}')
    print(f'Val loss: {val_losses[idx]}')
    print(f'Val acc: {val_accuracies[idx]}')
    if best_val_acc < val_accuracies[idx]:
        best_val_acc = val_accuracies[idx]
        best_epoch = idx + 1

