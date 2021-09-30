import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class DatasetAnalysisToolpack():
    
    def dataset_analysis_basic(self, df, df_title='', include_label_plot=False):
        
        print(f'{df_title} Analysis')
        print('=' * 10)
        
        print('First five rows')
        print(df.head(5))
        print()
        
        print('Object types')
        print(df.dtypes)
        print()
        
        print('Missing values')
        print(df.isnull().sum())
        print()
        
        print('Information')
        print(df.info())
        print()
        
        print('Categorical description')
        print(df.describe(include=[object]))
        print()
        
        try:
            print('Unique values')
            print(df['label'].value_counts())
        except:
            print('No label column for unique values.')
        
        if include_label_plot:
            try:
                df['label'].value_counts().plot(kind='bar')
                plt.xticks(rotation=0)
                plt.title(f'{df_title} Label')
                plt.xlabel('Label')
                plt.ylabel('Label Count')
                plt.savefig('files/fakenews_label.png',
                            dpi=1080,
                            format='png',
                            bbox_inches='tight')
                plt.show()
            except:
                print('No label column for plotting.')
        
        return None

class DatasetPartitioner():
    """DatasetPartitioner
    ==================
    Partition a large dataset into a list of smaller datasets.
    
    Parameters
    ==========
    frac: int, default=1
        A parameter in pd.Dataframe.sample
        Note: Must be less or equal to 1.
    
    n_sample: int, default=20000
        The length of the partitioned dataset.
        Note: Must be less than the length of the dataset.
    
    n_dataset: int, default=5
        The number of the partitioned dataset.
    
    random_state: int, default=42
        The random state of the pd.Dataframe.sample
    """
    def __init__(self, frac=1, n_sample=20000, n_dataset=5, random_state=42):
        self.frac = frac
        self.n_sample = n_sample
        self.n_dataset = n_dataset
        self.random_state = 42
        
        self.df_list = []
    
    def _check_partitioned(self):
        if len(self.df_list) == 0:
            return False
        else:
            return True
    
    def _autoscaling_by_n_sample(self, df, labelname):
        n_dataset_proper = int(min(df[labelname].value_counts()) / self.n_sample)
        
        if self.n_dataset > n_dataset_proper:
            print('Number of dataset autoscaled to fit the number of samples.')
            self.n_dataset = n_dataset_proper
        
        return self
    
    def _autoscaling_by_n_dataset(self, df, labelname):
        n_sample_proper = int(min(df[labelname].value_counts()) / self.n_dataset)
        
        if self.n_sample > n_sample_proper:
            print('Number of sample autoscaled to fit the number of dataset.')
            self.n_sample = n_sample_proper
        
        return self
    
    def _randomize_dataset(self, df):
        df = df.sample(frac=self.frac, random_state=self.random_state)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _partition_dataset(self, df):
        half = int(self.n_sample / 2)
        df_smaller = df.head(half)
        df.drop(range(half), axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df_smaller, df
    
    def _merge_df(self, df_list):
        df_new = pd.concat([df_list[0], df_list[1]], ignore_index=True)
        
        if len(df_list) > 2:
            for n in range(2, len(df_list)):
                df_new = pd.concat([df_new, df_list[n]], ignore_index=True)
        
        return df_new
    
    def __call__(self, df, labelname='label', n_label=2, autoscale='n_sample', shuffle=True):
        try:
            if(autoscale == 'n_sample'):
                self._autoscaling_by_n_sample(df, labelname)
            
            if(autoscale == 'n_dataset'):
                self._autoscaling_by_n_dataset(df, labelname)
            
            self.df_list = []
            
            df_list_bylabel = []
            
            for label in range(n_label):
                df_label = df[(df[labelname] == label)]
                df_label = self._randomize_dataset(df_label)
                df_list_bylabel.append(df_label)
            
            for n in range(self.n_dataset):
                df_small_list = []
                for label in range(n_label):
                    df_smaller, df_list_bylabel[label] = self._partition_dataset(df_list_bylabel[label])
                    df_small_list.append(df_smaller)
                
                df_new = self._merge_df(df_small_list)
                
                if shuffle:
                    df_new = self._randomize_dataset(df_new)
                
                self.df_list.append(df_new)
            
            return self.df_list
        except Exception as e:
            print('Something went wrong.')
            print(e)
            return None

class DatasetProcessor(Dataset):
    
    def __init__(self, texts1, texts2, labels, tokenizer, max_length):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, item):
        text1 = str(self.texts1[item])
        text2 = str(self.texts2[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text_one': text1,
            'text_two': text2,
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DataLoaderProcessor():
    
    def fakenews_dataset_loader(self, df, labels, tokenizer, max_length, batch_size):
        dataset = DatasetProcessor(
            texts1=df.title1_en.to_numpy(),
            texts2=df.title2_en.to_numpy(),
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )