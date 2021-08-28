# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import re
from time import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from FileManager import FileManager

#create FileManafer instance
fileManager = FileManager()

#establish test size
test_size = 0.2

#create TfidfVectorizer from sklearn.feature_extraction.text instances
vectorizer_one = TfidfVectorizer(stop_words='english')
vectorizer_two = TfidfVectorizer(stop_words='english')

def fakenews_train_test_split(df, test_size):
    X = df[['title1_en', 'title2_en']].to_numpy()
    y = df['label'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test

def vectorize_fakenews_data(X_train, X_test, vectorizer_one, vectorizer_two):
    X_train[:,0] = vectorizer_one.fit_transform(X_train[:,0])
    X_train[:,1] = vectorizer_two.fit_transform(X_train[:,1])
    
    X_test[:,0] = vectorizer_one.transform(X_test[:,0])
    X_test[:,1] = vectorizer_two.transform(X_test[:,1])
    
    return X_train, X_test

def model_training(text_clf, X_train, X_test, y_train):
    t0 = time()
    text_clf.fit(X_train, y_train)
    train_time = time() - t0
    
    t0 = time()
    test_pred = text_clf.predict(X_test)
    test_time = time() - t0
    
    return text_clf, test_pred, train_time, test_time

def train_model(X_train, X_test, y_train):
    text_clf_results = model_training(LinearSVC(loss='squared_hinge',
                                                penalty='l2', dual=False,
                                                tol=1e-3),
                                      X_train, X_test,y_train)
    return text_clf_results

def linear_SVC_Test_main():
    """load fakenews dataset"""
    df = fileManager.load_preprocessed_fakenews_dataset()
    
    X_train, X_test, y_train, y_test = fakenews_train_test_split(df, test_size)
    
    X_train, X_test = vectorize_fakenews_data(X_train, X_test, vectorizer_one, vectorizer_two)
    
    text_clf_results = train_model(X_train, X_test, y_train)
    
    print(text_clf_results)