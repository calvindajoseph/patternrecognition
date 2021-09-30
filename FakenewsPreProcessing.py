import re
from sklearn.preprocessing import LabelEncoder

from FileManager import FileManager
from DatasetClasses import DatasetPartitioner

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
    """Drop columns inside drop_column_names and drop all na values from dataframe df.
    
    Parameters
    ==========
    df: Pandas Dataframe
        The original Dataframe.
    
    drop_column_names: list of strings
        The column names to be dropped.
    
    Returns
    =======
    df: Pandas Dataframe
        The original Dataframe with the columns dropped and no na values.
    """
    df = df.drop(drop_column_names, axis=1)
    df = df.dropna()
    return df

def pre_process_columns_fakenews(df):
    """Drop special characters from title1_en and title2_en and encode label (0: related, 1: unrelated)
    
    Steps
    =====
    1)  Drop special characters in titles.
    2)  Drop html elements in titles.
    3)  Lowercase string in titles.
    4)  Replace all agreed and disagreed value in label to related.
    5)  Encode label column.
    
    Parameters
    ==========
    df: Pandas Dataframe
        The original Dataframe.
    
    Returns
    =======
    df: Pandas Dataframe
        The preprocessed Dataframe.
    """
    for title_en in ['title1_en', 'title2_en']:
        df[title_en] = df[title_en].map(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ', x))
        df[title_en] = df[title_en].map(lambda x: re.sub(r'<[^<]+?>', '', x))
        df[title_en] = df[title_en].map(lambda x: x.lower())
    
    df['label'] = df['label'].replace(to_replace = 'agreed', value = 'related')
    df['label'] = df['label'].replace(to_replace = 'disagreed', value = 'related')
    df['label'] = le.fit_transform(df[['label']])
    return df

# Load fakenews dataset.
df = fileManager.load_fakenews_dataset()

# Drop columns.
df = drop_columns(df, drop_column_names)

# Preprocess dataset.
df = pre_process_columns_fakenews(df)

# Save preprocessed dataset.
fileManager.save_preprocessed_fakenews_dataset(df)

### Partition Dataset ###

# Create DatasetPartitioner instance.
partitioner = DatasetPartitioner()

# Partition dataset.
small_datasets = partitioner(df)

# Save smaller datasets.
fileManager.save_preprocessed_fakenews_dataset_smaller(small_datasets)