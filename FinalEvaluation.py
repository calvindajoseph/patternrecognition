import warnings
warnings.filterwarnings("ignore")

from transformers import BertTokenizer

from sklearn.metrics import classification_report

import numpy as np

from ModelClasses import ModelClassifier
from DatasetClasses import DataLoaderProcessor
from FileManager import FileManager
import config

# Import Model
classifier = ModelClassifier()
   
# Create FileManager instance.
fileManager = FileManager()    
   
# Load dataframe.
df = fileManager.load_preprocessed_fakenews_dataset_smaller(file_number=2)

# Create DataLoaderProcessor instance.
data_loader = DataLoaderProcessor()

# Separate features from target.
X = df.drop(['label'], axis=1)
y = df.label.to_numpy()

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('./models/tokenizer/')

# Create data loader
test_data_loader = data_loader.fakenews_dataset_loader(X, y, tokenizer,config.MAX_LENGTH, config.BATCH_SIZE_FINAL_EVALUATION)

# Final Evaluation
y_true, y_pred, accuracy_per_batch = classifier.model_evaluation(test_data_loader)

print(f'Mean score: {np.mean(accuracy_per_batch)}')
print(f'Standard Deviation score: {np.std(accuracy_per_batch)}')

print(classification_report(y_true, y_pred, target_names=config.class_names))