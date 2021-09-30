import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

import transformers
from transformers import BertTokenizer

matplotlib.style.use('fivethirtyeight')

from FileManager import FileManager
from DatasetClasses import DatasetAnalysisToolpack

fileManager = FileManager()
analysis_toolpack = DatasetAnalysisToolpack()

df_preprocessed_fakenews = fileManager.load_preprocessed_fakenews_dataset()

df_smaller = fileManager.load_preprocessed_fakenews_dataset_smaller()

#analysis_toolpack.dataset_analysis_basic(df_preprocessed_fakenews, df_title='Preprocessed fakenews', include_label_plot=True)
#analysis_toolpack.dataset_analysis_basic(df_smaller, df_title='Smaller', include_label_plot=True)

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []

def get_token_lengths(df_series):
    token_lens = []
    
    for title in df_series:
        tokens = tokenizer.encode(title, max_length=512)
        token_lens.append(len(tokens))
    
    return token_lens

token_lens = get_token_lengths(df_preprocessed_fakenews.title1_en)
token_lens_np = np.array(token_lens)

print(f'Mean\t{np.mean(token_lens_np)}')
print(f'Median\t{np.median(token_lens_np)}')
print(f'Min\t{np.min(token_lens_np)}')
print(f'Max\t{np.max(token_lens_np)}')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

sns.displot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()