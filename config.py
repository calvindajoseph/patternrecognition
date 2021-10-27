"""
Model configuration.

CAUTION: Changing parameters may break the code.
"""

# Class names
class_names = ['related', 'unrelated']

# Set random state.
RANDOM_STATE = 42

# Set the pretrained vocabulary tokenizer from Huggingface library.
VOCAB_NAME = 'bert-base-uncased'

# Set the pretrained BERTModel from Huggingface library.
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# Model filename
MODEL_DIR = './models/model_epochs19Oct/model_state_1.pth'

# Set DataLoader hyperparameters
MAX_LENGTH = 80
BATCH_SIZE_TRAINING = 16
BATCH_SIZE_VALIDATION = 256
BATCH_SIZE_TESTING = 256

BATCH_SIZE_FINAL_EVALUATION = 1000

# Set model hyperparameters
LEARNING_RATE=2e-5
EPOCHS = 3