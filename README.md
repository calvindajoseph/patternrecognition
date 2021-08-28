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

### Update Anaconda
On Windows, to update Anaconda open Anaconda Prompt. Then type in:
> $ conda update --all

### Update/upgrade to Spyder 5.0.5
For updating spyder, after updating Anaconda, type in (in Anaconda Prompt):
> $ conda install spyder=5.0.5

### Before running any code
Please unzip the fake-news-pair-classification-challenge.zip and name the train.csv to fake-news-pair-classification-challenge.csv in the files folder

## Pre-processing
The WDSM 2019 dataset contained eight columns. We dropped five and kept three:
1. title1_en
2. title2_en
3. label

Then, we changed 'agreed' and 'disagreed' text from the label column with related. There are two classes left in the label:
1. 'related'
2. 'unrelated'

And encode them to:
'related' to 0s
'unrelated' to 1s

From title1_en and title2_en, we deleted every special characters and html elements.

## Future Plans
According to our research, pretrained BERT is the most efficient machine learning method that has been conducted to the dataset.
