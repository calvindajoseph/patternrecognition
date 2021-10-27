"""
Manages our file.

This class is responsible to import and export files.
"""

# Import io module for input output
import io

# Import pandas
import pandas as pd

# Import BertTokenizer
from transformers import BertTokenizer

class FileManager:
    """
    The filemanager.
    
    The attributes are directories of various datasets and tokens.
    """
    
    def __init__(self):
        self.folder_dataset = 'datasets'
        self.folder_tokenizer = 'bert_tokenizer'
        self.folder_dataset_preprocessed_fakenews_smaller = 'datasets/preprocessed_fakenews_smaller'
        
        self.filename_fakenews_dataset = 'fake_news_classification_challenge.csv'
        
        self.filename_preprocessed_fakenews = 'preprocessed_fakenews.csv'
        self.filename_preprocessed_fakenews_smaller = ['preprocessed_fakenews_smaller_1.csv',
                                                       'preprocessed_fakenews_smaller_2.csv',
                                                       'preprocessed_fakenews_smaller_3.csv',
                                                       'preprocessed_fakenews_smaller_4.csv',
                                                       'preprocessed_fakenews_smaller_5.csv']
        
        self.filename_fakenews_dataset_analysis = 'fake_news_classification_analysis.txt'
        
        self.filename_preprocessed_fakenews_analysis = 'preprocessed_fakenews_analysis.txt'
        
        self.dir_tokenizer = 'models/tokenizer/'
        
    
    def import_csv_to_dataframe(self, foldername, filename):
        """Load a pandas dataframe
        
        Parameters
        =========
        foldername: string
            The foldername of the folder that contains the dataset.
        
        filename: string
            The filename of the csv file.
        
        Returns
        =======
        df: Dataframe
            The dataframe.
        """
        directory = foldername + '/' + filename
        
        try:
            df = pd.read_csv(directory)
            print("Dataframe imported from " + directory)
            return df
        except:
            print("Dataframe could not load.")
            return None
    
    def export_dataframe_to_csv(self, df, foldername, filename):
        """Save a pandas dataframe
        
        Parameters
        =========
        
        df: Dataframe
            The dataframe to be saved.
        
        foldername: string
            The foldername of the folder that contains the dataset.
        
        filename: string
            The filename of the csv file.
        """
        directory = foldername + '/' + filename
        
        try:
            df.to_csv(directory, index=False)
            print("Dataframe exported to " + directory)
        except:
            print("Dataframe could not be exported.")
        
        return None
    
    def export_txt_file(self, content, foldername, filename):
        """Save a text file
        
        Parameters
        =========
        
        content: string
            The text.
        
        foldername: string
            The foldername of the folder that contains the dataset.
        
        filename: string
            The filename of the csv file.
        """
        directory = foldername + '/' + filename
        
        try:
            f = io.open(directory, 'wt', encoding='utf-8')
            f.write(content)
            f.close()
        except:
            print("Could not export file:", filename)
        
        return None
    
    def load_fakenews_dataset(self, foldername=None):
        """Load fakenews dataset from dataset folder to a Pandas Dataframe.
        
        Parameters
        =========
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the fakenews dataset.
        """
        if foldername:
            df = self.import_csv_to_dataframe(foldername, self.filename_fakenews_dataset)
        else:
            df = self.import_csv_to_dataframe(self.folder_dataset, self.filename_fakenews_dataset)
        return df
    
    def load_preprocessed_fakenews_dataset(self, foldername=None):
        """Load preprocessed fakenews dataset from dataset folder to a Pandas Dataframe.
        
        Parameters
        =========
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the preprocessed fakenews dataset.
        """
        if foldername:
            df = self.import_csv_to_dataframe(foldername, self.filename_preprocessed_fakenews)
        else:
            df = self.import_csv_to_dataframe(self.folder_dataset, self.filename_preprocessed_fakenews)
        return df
    
    def load_preprocessed_fakenews_dataset_smaller(self, file_number=1, filename=None, foldername=None):
        """Load small preprocessed fakenews dataset from dataset/preprocessed_fakenews_smaller folder to a Pandas Dataframe.
        
        Parameters
        =========
        file_number: int, default=1
            The file number of the smaller dataset.
        
        filename: string, default=None
            The filename of the smaller datset.
        
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the preprocessed fakenews dataset.
        """
        if filename == None:
            filename = self.filename_preprocessed_fakenews_smaller[file_number-1]
        
        if foldername:
            df = self.import_csv_to_dataframe(foldername, filename)
        else:
            df = self.import_csv_to_dataframe(self.folder_dataset_preprocessed_fakenews_smaller, filename)
        return df
    
    def load_tokenizer(self, token_dir=None):
        """Load BERT tokenizer from directory.
        
        Parameters
        =========
        token_dir: string, default=None
            The directory of the tokenizer.
        
        Returns
        =======
        tokenizer: BertTokenizer
            The BertTokenizer.
        """
        if token_dir:
            tokenizer = BertTokenizer.from_pretrained(token_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.dir_tokenizer)
        return tokenizer
    
    def save_preprocessed_fakenews_dataset(self, df, foldername=None, filename=None):
        """Save preprocessed fakenews dataset to dataset folder as csv.
        
        Parameters
        =========
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the preprocessed fakenews dataset.
        """
        if foldername == None:
            foldername = self.folder_dataset
        
        if filename == None:
            filename = self.filename_preprocessed_fakenews
        
        self.export_dataframe_to_csv(df, foldername, filename)
        return None
    
    def save_preprocessed_fakenews_dataset_smaller(self, df_list, foldername=None):
        """Save preprocessed fakenews dataset to dataset folder as csv.
        
        Parameters
        =========
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the preprocessed fakenews dataset.
        """
        if foldername == None:
            foldername = self.folder_dataset_preprocessed_fakenews_smaller
        
        for i in range(int(len(df_list))):
            filename = f'preprocessed_fakenews_smaller_{i+1}.csv'
            self.export_dataframe_to_csv(df_list[i], foldername, filename)
        
        return None
    
    def save_tokenizer(self, tokenizer, foldername=None, filename=None):
        """Save tokenizer to directory.
        
        Parameters
        =========
        foldername: string, default=None
            The foldername of the folder that contains the dataset.
        
        Returns
        =======
        df: Dataframe
            The dataframe of the preprocessed fakenews dataset.
        """
        if foldername == None:
            foldername = self.folder_dataset_preprocessed_fakenews_smaller
    
    def save_fakenews_dataset_analysis(self, content):
        """
        Unused. Usage TBA.
        """
        self.export_txt_file(self.path, content, self.filename_fakenews_dataset_analysis)
        return None
    
    def save_preprocessed_fakenews_analysis(self, content):
        """
        Unused. Usage TBA.
        """
        self.export_txt_file(self.path, content, self.filename_preprocessed_fakenews_analysis)
        return None