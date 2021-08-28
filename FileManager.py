# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np

class FileManager:
    
    def __init__(self):
        self.path = r'C:\Users\calda\OneDrive\Documents\University - UC Master\Semester 2 2021\11512 Pattern Recognition and Machine Learning PG\Assignment 1\AssignmentTest\files'
        
        self.filename_fakenews_dataset = 'fake_news_classification_challenge.csv'
        
        self.filename_preprocessed_fakenews = 'preprocessed_fakenews.csv'
        
    
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
    
    def load_fakenews_dataset(self):
        df = self.import_csv_to_dataframe(self.path, self.filename_fakenews_dataset)
        return df
    
    def load_preprocessed_fakenews_dataset(self):
        df = self.import_csv_to_dataframe(self.path, self.filename_preprocessed_fakenews)
        return df
    
    def save_preprocessed_fakenews_dataset(self, df):
        df = self.export_csv_to_dataframe(df, self.path, self.filename_preprocessed_fakenews)
        return None
    