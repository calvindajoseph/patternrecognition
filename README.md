# Pattern Recognition and Machine Learning Project

## Project Name: Finding relationship between two texts?

This project focuses on building a machine learning model that can determine whether two texts are related or unrelated.

## Installation

The dataset can be obtained from:
> https://www.kaggle.com/c/fake-news-pair-classification-challenge

The code were written in Spyder 5. To install spyder, simply download Anaconda.

### Update Anaconda
On Windows, to update Anaconda open Anaconda Prompt. Then type in:
> $ conda update --all

### Update/upgrade to Spyder 5.0.5
For updating spyder, after updating Anaconda, type in (in Anaconda Prompt):
> $ conda install spyder=5.0.5

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
