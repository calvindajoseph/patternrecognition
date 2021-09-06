# -*- coding: utf-8 -*-
import os
import io

import pandas as pd
import numpy as np

class FileManager:
    
    def __init__(self):
        self.path = r'C:\Users\calda\OneDrive\Documents\University - UC Master\Semester 2 2021\11512 Pattern Recognition and Machine Learning PG\Assignment 1\AssignmentTest\files'
        
        self.filename_fakenews_dataset = 'fake_news_classification_challenge.csv'
        
        self.filename_preprocessed_fakenews = 'preprocessed_fakenews.csv'
        
        self.filename_fakenews_dataset_analysis = 'fake_news_classification_analysis.txt'
        
        self.filename_preprocessed_fakenews_analysis = 'preprocessed_fakenews_analysis.txt'
        
    
    def import_csv_to_dataframe(self, path, filename):
        path = path + '\\' + filename
        
        try:
            df = pd.read_csv(path)
            print("Dataframe imported from " + path)
            return df
        except:
            print("Dataframe could not load.")
            return None
    
    def export_csv_to_dataframe(self, df, path, filename):
        path = path + '\\' + filename
        
        try:
            df.to_csv(path, index=False)
            print("Dataframe exported to " + path)
        except:
            print("Dataframe could not be exported.")
        
        return None
    
    def export_txt_file(self, path, content, filename):
        path = path + '\\' + filename
        
        try:
            f = io.open(path, 'wt', encoding='utf-8')
            f.write(content)
            f.close()
        except:
            print("Could not export file:", filename)
        
        return None
    
    def load_fakenews_dataset(self):
        df = self.import_csv_to_dataframe(self.path, self.filename_fakenews_dataset)
        return df
    
    def load_preprocessed_fakenews_dataset(self):
        df = self.import_csv_to_dataframe(self.path, self.filename_preprocessed_fakenews)
        return df
    
    def save_preprocessed_fakenews_dataset(self, df):
        df = self.export_csv_to_dataframe(df, self.path, self.filename_preprocessed_fakenews)
        return None
    
    def save_fakenews_dataset_analysis(self, content):
        self.export_txt_file(self.path, content, self.filename_fakenews_dataset_analysis)
        return None
    
    def save_preprocessed_fakenews_analysis(self, content):
        self.export_txt_file(self.path, content, self.filename_preprocessed_fakenews_analysis)
        return None