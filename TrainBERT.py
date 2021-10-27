# Import transformer from Huggingface library
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# Import pytorch modules
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Import train_test_split from the sklearn
from sklearn.model_selection import train_test_split

# Import numpy
import numpy as np

# Import other modules for training tracking
from collections import defaultdict
import time
from datetime import timedelta

# Import local modules
from FileManager import FileManager
from DatasetClasses import DataLoaderProcessor
from ModelClasses import SequenceClassifier

import config

# Create FileManager instance.
fileManager = FileManager()

# Create DataLoaderProcessor instance.
data_loader = DataLoaderProcessor()

# Class names
class_names = config.class_names

# Load dataframe.
df = fileManager.load_preprocessed_fakenews_dataset_smaller(file_number=1)

# Set random state.
RANDOM_STATE = config.RANDOM_STATE

# Set the pretrained vocabulary tokenizer from Huggingface library.
VOCAB_NAME = config.VOCAB_NAME

# Set the pretrained BERTModel from Huggingface library.
PRE_TRAINED_MODEL_NAME = config.PRE_TRAINED_MODEL_NAME

# Set model folder
model_folder = 'model_epochs_30Sept_ver2'

# Set DataLoader hyperparameters
MAX_LENGTH = config.MAX_LENGTH
BATCH_SIZE_TRAINING = config.BATCH_SIZE_TRAINING
BATCH_SIZE_VALIDATION = config.BATCH_SIZE_VALIDATION
BATCH_SIZE_TESTING = config.BATCH_SIZE_TESTING

# Set model hyperparameters
LEARNING_RATE=config.LEARNING_RATE
EPOCHS = config.EPOCHS

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

# Load data to torch DataLoader
train_data_loader = data_loader.fakenews_dataset_loader(X_train, y_train, tokenizer,MAX_LENGTH, BATCH_SIZE_TRAINING)
val_data_loader = data_loader.fakenews_dataset_loader(X_val, y_val, tokenizer,MAX_LENGTH, BATCH_SIZE_VALIDATION)
test_data_loader = data_loader.fakenews_dataset_loader(X_test, y_test, tokenizer,MAX_LENGTH, BATCH_SIZE_TESTING)

# Establish a model
model = SequenceClassifier(n_classes=len(class_names), pretrained_model=PRE_TRAINED_MODEL_NAME)
# Load model to device (If no GPU, then CPU)
model = model.to(device)

# Set optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
# Set total steps
total_steps = len(train_data_loader) * EPOCHS

# Set scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Set the loss function to device
loss_fn = nn.CrossEntropyLoss().to(device)

# Train epoch
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, n_epochs):
    """
    Train one epoch.
    """
    
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
    """
    Model evaluation. For validation set and test set.
    """
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

# Set logs for tracking
logs_str = f'Learning Rate: {config.LEARNING_RATE}'
logs_str = logs_str + f'\nEpoch: {config.EPOCHS}'
logs_str = logs_str + f'\nTrain Batch Size: {config.BATCH_SIZE_TRAINING}'
logs_str = logs_str + '\n\nModel Training'
logs_str = logs_str + '\n=============='

# Save losses and accuracies to a list
train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

# Set training start time
training_time_start = time.time()

# Main training loop
for epoch in range(EPOCHS):
    
    epoch_time_start = time.time()
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    logs_str = logs_str + f'\n\nEpoch {epoch + 1}'
    logs_str = logs_str + '\n======='
    
    train_acc, train_loss = train_epoch(model, train_data_loader,
                                        loss_fn, optimizer, device,
                                        scheduler, len(X_train), epoch)
    print(f'Train loss: {train_loss}; Accuracy: {train_acc}')
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    logs_str = logs_str + f'\n\nTrain loss: {train_loss}'
    logs_str = logs_str + f'\nTrain acc: {train_acc}'
    
    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn,
                                   device, len(X_val))
    
    print(f'Val loss: {val_loss}; Accuracy: {val_acc}')
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    logs_str = logs_str + f'\nVal loss: {val_loss}'
    logs_str = logs_str + f'\nVal acc: {val_acc}'
    
    filename = f'models/{model_folder}/model_state_{epoch+1}.pth'
    torch.save(model.state_dict(), filename)
    
    epoch_time_elapsed = time.time() - epoch_time_start
    print(f'Epoch runtime {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')
    logs_str = logs_str + f'\n\nEpoch runtime: {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s'

# Print out the training time
# Usually about 4-7 hours
train_time_elapsed = time.time() - training_time_start
train_time_delta = timedelta(seconds=int(train_time_elapsed))
print(f'Training runtime: {train_time_delta}')
logs_str = logs_str + f'\n\nTraining runtime: {train_time_delta}'

# Set up for testing with the last epoch
testing_time_start = time.time()
test_acc, test_loss = eval_model(model, test_data_loader, loss_fn,
                                 device, len(X_test))

# Print out testing results
print(f'Test loss: {test_loss}; Accuracy: {test_acc}')
test_time_elapsed = time.time() - testing_time_start
test_time_delta = timedelta(seconds=int(test_time_elapsed))
print(f'Testing runtime: {test_time_delta}')
logs_str = logs_str + f'\n\nTest loss: {test_loss}; Accuracy: {test_acc}'
logs_str = logs_str + f'\n\nTesting runtime: {test_time_delta}'

# Find the best validation accuracy
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

# Set logs filename
filename = f'models/{model_folder}/logs.txt'

# Save logs to a txt file
f = open(filename, 'w')
f.write(logs_str)