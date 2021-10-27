"""
A code to analyse the lengths of sentences in our dataset.

Since our model takes a fixed amount of tokens, we
need to decide what is our maximum length.
"""

# Import numpy
import numpy as np

# Import matplotlib modules
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

# Import tokenizer
from transformers import BertTokenizer

# Set plt style
matplotlib.style.use('fivethirtyeight')

# Import FileManager
from FileManager import FileManager
# Import DatasetAnalysisToolpack
from DatasetClasses import DatasetAnalysisToolpack

# Set filemanager instance
fileManager = FileManager()
# Set analysis toolpack instance
analysis_toolpack = DatasetAnalysisToolpack()

# Import the raw dataset
df_preprocessed_fakenews = fileManager.load_preprocessed_fakenews_dataset()

# Import a smaller dataset
df_smaller = fileManager.load_preprocessed_fakenews_dataset_smaller()

#analysis_toolpack.dataset_analysis_basic(df_preprocessed_fakenews, df_title='Preprocessed fakenews', include_label_plot=True)
#analysis_toolpack.dataset_analysis_basic(df_smaller, df_title='Smaller', include_label_plot=True)

# Set the tokenizer mode
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
# Set the tokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# List to hold the lengths of each tokens
token_lens = []

def get_token_lengths(df_series):
    """
    Get a list of token lengths
    
    Parameters
    ==========
    df_series: Pandas Dataframe
        The dataset for getting the tokens
    
    Returns
    =======
    token_lens: list
        A list that has the length of each token.
    """
    token_lens = []
    
    for title in df_series:
        tokens = tokenizer.encode(title, max_length=512)
        token_lens.append(len(tokens))
    
    return token_lens

# Get all token lengths
token_lens = get_token_lengths(df_preprocessed_fakenews.title1_en)
# Convert to numpy array
token_lens_np = np.array(token_lens)

# Print the summary of the token lengths
print(f'Mean\t{np.mean(token_lens_np)}')
print(f'Median\t{np.median(token_lens_np)}')
print(f'Min\t{np.min(token_lens_np)}')
print(f'Max\t{np.max(token_lens_np)}')

# Set seaborn styles
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Set the colours 
COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

# Set the colours
sns.set_palette(sns.color_palette(COLORS_PALETTE))

# Plot the token lengths
sns.displot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()