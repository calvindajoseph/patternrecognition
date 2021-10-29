# Pattern Recognition and Machine Learning Project

## Project Name: Finding relationship between two texts?

This project focuses on building a machine learning model that can determine whether two texts are related or unrelated.

## Installation

The dataset can be obtained from:
> https://www.kaggle.com/c/fake-news-pair-classification-challenge

There is a bug in Github Desktop for Windows that will not allow you to clone repositories. If this happens, try the following:
1. On the top left corner, click File
2. Select Options...
3. Sign out from your account
4. Sign back in with your username and password
5. Try to clone the repo again

The code were written in Spyder 5. To install spyder, simply download Anaconda.

### Libraries
Please install all modules. Otherwise none of the code would run.

Scikit learn. Download: https://scikit-learn.org/stable/install.html
Pytorch. Download: https://pytorch.org/
Huggingface Library: https://huggingface.co/transformers/installation.html

### Update Anaconda
On Windows, to update Anaconda open Anaconda Prompt. Then type in:
> $ conda update --all

### Update/upgrade to Spyder 5.0.5
For updating spyder, after updating Anaconda, type in (in Anaconda Prompt):
> $ conda install spyder=5.0.5

### Before running any code

1. Please unzip the fake-news-pair-classification-challenge.zip and name the train.csv to fake-news-pair-classification-challenge.csv in the files folder.
2. Please ensure the libraries are downloaded.
3. The full model is available here: https://drive.google.com/drive/folders/1UQh55mVcztmEB2JEexUOxrrQSZrYDWmG?usp=sharing

* Note: The model is too big for our Github repo, so please download our model from the Google Drive link.

## Code Structure
The majority of the code was written with object oriented programming.

There are main files for each task, such as FakenewsPreProcessing.py was the main file for the Preprocessing stage.

Supporting files are mainly for testing and generating plots for the report.

List of supporting files:

1. ClassifierExample.py is to demonstrate how the model can be used to classify raw text.
2. ClassifierGUI.py has a tkinter demonstration of the model.
3. DataAnalysis.py was used to determine the maximum size of each feature. Since we are goint to user deep learning, we need the exact number of input tokens for each datapoints. This file analyse the distribution of text length and determine the length.
4. ValAccuracyGraph.py is to generate the plot of validation accuracy vs epoch.

Lastly, there are python files that contains only classes. This chapter will describe the classes file below.

### FileManager.py
The FileManager manages all files. It is responsible for any major input and output process.

### DatasetClasses.py
Any dataset related classes are in this file.

### ModelClasses.py
Any model related classes are in this file.

## General Flow
Please follow the guidelines to know the chronology of the project.

### Preprocessing
Main file: FakenewsPreProcessing.py

The WDSM 2019 dataset contained eight columns. We dropped five and kept three:
1. title1_en
2. title2_en
3. label

Then, we changed 'agreed' and 'disagreed' text from the label column with related. There are two classes left in the label:
1. 'related'
2. 'unrelated'

And encode them to:
1. 'related' to 0s
2. 'unrelated' to 1s

From title1_en and title2_en, we deleted every special characters and html elements.

The dataset was partitioned due to lack of resources. We divide the dataset into five stratified smaller datasets. Each smaller dataset has 20,000 samples, 50% labelled related and 50% labelled unrelated.

### Feature Extraction and Model Training
Main file: TrainBERT.py

We are using BERT model from Huggingface Library, and pytorch as a basis.

The dataset is the preprocessed_fakenews_smaller_1.csv
Please run FakenewsPreProcessing.py to get this file.

Steps:
1. Load dataset into pytorch Dataset.
2. Tokenize the features using BertTokenizer.
3. Split the dataset into train test validation set.
4. Split each set into batches with pytorch DataLoader.
5. Fine Tune BertModel with training set.
6. For each epoch, find the accuracy and loss of the model on validation set.
7. Save the model.
8. Evaluate the model on testing set.
9. Save the logs into a text file.

### Final Evaluation
Main file: FinalEvaluation.py

Since we have 5 different datasets, we decided to use one of them as a testing set for further model evaluation. The final evaluation has 20,000 datapoints.

The final evaluation in the report came from this process.
