# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import re
from sklearn.preprocessing import LabelEncoder

from FileManager import FileManager

#create FileManafer instance
fileManager = FileManager()

#list of the dropeed column names
drop_column_names = ['id',
                     'tid1',
                     'tid2',
                     'title1_zh',
                     'title2_zh']

#create LabelEncoder from sklearn.preprocessing instance
le = LabelEncoder()

def drop_columns(df, drop_column_names):
    """Drop columns inside drop_column_names from dataframe df"""
    df = df.drop(drop_column_names, axis=1)
    df = df.dropna()
    return df

def pre_process_columns_fakenews(df):
    """Drop special characters from title1_en and title2_en and encode label (0:agreed, 1:disagreed, 2:unrelated)"""
    df['title1_en'] = df['title1_en'].map(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ', x))
    df['title1_en'] = df['title1_en'].map(lambda x: re.sub(r'<[^<]+?>', '', x))
    df['title1_en'] = df['title1_en'].map(lambda x: x.lower())
    
    df['title2_en'] = df['title2_en'].map(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ', x))
    df['title2_en'] = df['title2_en'].map(lambda x: re.sub(r'<[^<]+?>', '', x))
    df['title2_en'] = df['title2_en'].map(lambda x: x.lower())
    
    df['label'] = df['label'].replace(to_replace = 'agreed', value = 'related')
    df['label'] = df['label'].replace(to_replace = 'disagreed', value = 'related')
    df['label'] = le.fit_transform(df[['label']])
    return df

def preprocessing_main():
    """preprocess the fakenews dataset"""
    
    """load fakenews dataset"""
    df = fileManager.load_fakenews_dataset()
    
    """drop all columns except english titles and label"""
    df = drop_columns(df, drop_column_names)
    
    """preprocess dataset including:
        removing special characters
        encode label column (0:agreed, 1:disagreed, 2: unrelated)"""
    df = pre_process_columns_fakenews(df)
    
    """save the dataset"""
    fileManager.save_preprocessed_fakenews_dataset(df)
    
    return None


