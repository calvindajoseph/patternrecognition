# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib

matplotlib.style.use('fivethirtyeight')

from FileManager import FileManager

fileManager = FileManager()

def create_fakenews_analysis():
    df = fileManager.load_fakenews_dataset()
    
    content = 'Fakenews Dataset Analysis\n'
    
    df_head = str(df.head(5))
    content = content + '\nDataset First Five Rows\n' + df_head
    print(df_head)
    
    content = content + '\n'
    df_dtypes = str(df.dtypes)
    content = content + '\nDataset Object Types\n' + df_dtypes
    print(df_dtypes)
    
    content = content + '\n'
    df_isnull = str(df.isnull().sum())
    content = content + '\nDataset Missing Values\n' + df_isnull
    print(df_isnull)
    
    content = content + '\n'
    df_info = str(df.info())
    content = content + '\nDataset Information\n' + str(df.info())
    print(df_info)
    
    content = content + '\n'
    df_describe = str(df.describe(include=[object]))
    content = content + '\nDataset Categorical Description\n' + df_describe
    print(df_describe)
    
    content = content + '\n'
    df_label_value_counts = str(df['label'].value_counts())
    content = content + '\nDataset Label Value Counts\n' + df_label_value_counts
    print(df_label_value_counts)
    
    fileManager.save_fakenews_dataset_analysis(content)
    
    df['label'].value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.title('Fakenews Dataset Label')
    plt.xlabel('Label')
    plt.ylabel('Label Count')
    plt.savefig('files/fakenews_label.png',
                dpi=1080,
                format='png',
                bbox_inches='tight')
    plt.show()
    
    return None

def create_preprocessed_fakenews_analysis():
    df = fileManager.load_preprocessed_fakenews_dataset()
    
    content = 'Fakenews Dataset Analysis\n'
    
    df_head = str(df.head(5))
    content = content + '\nDataset First Five Rows\n' + df_head
    print(df_head)
    
    content = content + '\n'
    df_dtypes = str(df.dtypes)
    content = content + '\nDataset Object Types\n' + df_dtypes
    print(df_dtypes)
    
    content = content + '\n'
    df_isnull = str(df.isnull().sum())
    content = content + '\nDataset Missing Values\n' + df_isnull
    print(df_isnull)
    
    content = content + '\n'
    df_info = str(df.info())
    content = content + '\nDataset Information\n' + str(df.info())
    print(df_info)
    
    content = content + '\n'
    df_describe = str(df.describe(include=[object]))
    content = content + '\nDataset Categorical Description\n' + df_describe
    print(df_describe)
    
    content = content + '\n'
    df_label_value_counts = str(df['label'].value_counts())
    content = content + '\nDataset Label Value Counts\n' + df_label_value_counts
    print(df_label_value_counts)
    
    fileManager.save_preprocessed_fakenews_analysis(content)
    
    df['label'].value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.title('Fakenews Dataset Label')
    plt.xlabel('Label')
    plt.ylabel('Label Count')
    plt.savefig('files/preprocessed_fakenews_label.png',
                dpi=1080,
                format='png',
                bbox_inches='tight')
    plt.show()
    
    return None

def run_data_analysis():
    create_fakenews_analysis()
    create_preprocessed_fakenews_analysis()
    
    return None

run_data_analysis()