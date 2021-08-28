# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:40:40 2021

@author: calda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from FileManager import FileManager

fileManager = FileManager()

def df_information(df):
    print("Fakenews dataset info:")
    print(df.info())
    df_info = df.info()
    print("\nFakenews dataset null values:")
    print(df.isnull().sum())
    df_null_values = df.isnull().sum()
    return df_info, df_null_values
    
def visualization_main():
    df = fileManager.load_fakenews_dataset()
    df_info, df_null_values = df_information(df)
    print(df_info)
    return None