import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

import torch


class Dataset:
    """
        Using HuggingFace to download the dataset
    """

    def loader( self, file_path ):

        data = pd.read_csv(file_path, sep='\t')
        return data.iloc[:, 0], data.iloc[:, 1]
    
    def label_transform( self, y_train, y_val, y_test ):

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        return y_train, y_val, y_test


    def load_dataset(self, train_file_path, val_file_path, test_file_path):
        """
            This function loads the dataset
        """

        self.X_train, y_train = self.loader( train_file_path )
        self.X_val, y_val = self.loader( val_file_path )
        self.X_test, y_test = self.loader( test_file_path )

        self.y_train, self.y_val, self.y_test = self.label_transform( y_train, y_val, y_test )
        

class CharTokenizer:
    def __init__(self, vocab, pad_token_id=0):
        self.stoi = vocab["stoi"]
        self.pad_token_id = pad_token_id

    def __call__(self, text, max_length=256, padding=True, truncation=True):

        ids = [self.stoi.get(c, 0) for c in text]

        if truncation:
            ids = ids[:max_length]

        if padding:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        attention_mask = [1 if i != self.pad_token_id else 0 for i in ids]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


class Tokenization:
    
    def __init__(self, train_file_path, val_file_path, test_file_path, batch_size, max_length):
        
        # self.tokenizer = AutoTokenizer.from_pretrained( model_name, trust_remote_code=True )

        meta = self.load_pkl( "misc/meta.pkl" )
        self.tokenizer = CharTokenizer(vocab = meta) 

        self.train_file_path = train_file_path 
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path
        self.batch_size = batch_size
        self.max_length = max_length
    

    def load_pkl(self, dir):
        with open(dir, 'rb') as f:
            meta = pickle.load(f)
        return meta
    
    def tokenize_inputs(self, texts):

        encodings = [self.tokenizer(t, padding=True, truncation=True, max_length=self.max_length) for t in texts]

        input_ids = torch.stack([e["input_ids"] for e in encodings])
        attention_mask = torch.stack([e["attention_mask"] for e in encodings])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def encode( self, X, y ):
        return self.tokenize_inputs(X.tolist()), torch.tensor(y)
    
    def tensor_conversion( self, X_encoding, y_encoding ):
        return TensorDataset(X_encoding['input_ids'], X_encoding['attention_mask'], y_encoding)
    
    def convert_to_dataloader( self, dataset ):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    
    def process_dataset( self ):

        dataset = Dataset()

        dataset.load_dataset( train_file_path=self.train_file_path, val_file_path=self.val_file_path, test_file_path=self.test_file_path )

        print( "[Done] Dataset Loaded!" )

        train_encodings, y_train_tensor = self.encode( dataset.X_train, dataset.y_train )
        val_encodings, y_dev_tensor = self.encode( dataset.X_val, dataset.y_val )
        test_encodings, y_test_tensor = self.encode( dataset.X_test, dataset.y_test )

        train_dataset = self.tensor_conversion( train_encodings, y_train_tensor )
        val_dataset = self.tensor_conversion( val_encodings, y_dev_tensor )
        test_dataset = self.tensor_conversion( test_encodings, y_test_tensor )

        train_loader = self.convert_to_dataloader(train_dataset)
        dev_loader = self.convert_to_dataloader(val_dataset)
        test_loader = self.convert_to_dataloader(test_dataset)

        return train_loader, dev_loader, test_loader
        
                
def dataset_loader( dataset_path, test_path, batch_size, max_length ):

    train_path = dataset_path + "train.tsv"
    val_path = dataset_path + "val.tsv"

    tokenization = Tokenization(train_path, val_path, test_path, batch_size, max_length)

    return tokenization.process_dataset()