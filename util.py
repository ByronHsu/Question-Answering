import pandas as pd
from os.path import join
from transformers import *
import torch
import torch.utils.data as data

# =====================
# For Training
# =====================
def parse_xlsx(path):
    '''
    Parse xlsx file
    Return:
        all_senA (list[string]): Questions
        all_senB (list[string]): Anwsers
        all_isNext (list[int]): Whether B is the next sentence of A
    '''
    df = pd.read_excel(path)
    all_senA = []
    all_senB = []
    all_isNext = []

    for i in range(df.shape[0]):
        article = df.iloc[i, 1]
        question = df.iloc[i, 2]
        right_idx = df.iloc[i, 7]

        for j in range(4):
            choice = df.iloc[i, 3 + j]
            all_senA.append(article + question)
            all_senB.append(choice)
            # 0 indicates sequence B is a continuation of sequence A, 1 indicates sequence B is a random sequence.
            if j + 1 == right_idx:
                all_isNext.append(0)
            else:
                all_isNext.append(1)


    return all_senA, all_senB, all_isNext

def custom_collate(tokenizer):
    def callback(batch):
        zipped = list(zip(*batch))
        batch_sen_pair = list(zip(zipped[0], zipped[1]))
        batch_isNext = zipped[2]
        max_length = min(max([len(p[0]) + len(p[1]) for p in batch_sen_pair]), 512)
        batch_input_ids = []
        batch_token_type_ids = []
        for sen_pair in batch_sen_pair:
            out = tokenizer.encode_plus(*sen_pair, max_length = max_length, pad_to_max_length = True)
            batch_input_ids.append(out['input_ids'])
            batch_token_type_ids.append(out['token_type_ids'])
        return torch.tensor(batch_input_ids), torch.tensor(batch_token_type_ids), torch.tensor(batch_isNext)
    return callback

class Dataset(data.Dataset):
    def __init__(self, tokenizer):
        self.array = self._parse_data()
        self.tokenizer = tokenizer
        print('finish parsing data')
        
    def _parse_data(self):
        array = []
        for i in range(6):
            senA, senB, isNext = parse_xlsx(join('data', '{}.xlsx'.format(i + 1)))
            array += list(zip(senA, senB, isNext))
        return array

    def __getitem__(self, i):
        item = self.array[i]
        return item
    
    def __len__(self):
        return len(self.array)

# =====================
# For Testing
# =====================
def test_parse_xlsx(path):
    '''
    Parse xlsx file
    Return:
        all_senA (list[string]): Questions
        all_senB (list[string]): Anwsers
        all_id (list(int)): The id of testing row
    '''
    df = pd.read_excel(path)
    all_senA = []
    all_senB = []
    all_id = []

    for i in range(df.shape[0]):
        _id = df.iloc[i, 0]
        article = df.iloc[i, 1]
        question = df.iloc[i, 2]

        for j in range(4):
            choice = df.iloc[i, 3 + j]
            all_senA.append(article + question)
            all_senB.append(choice)
            all_id.append(_id)

    return all_senA, all_senB, all_id

def test_custom_collate(tokenizer):
    def callback(batch):
        zipped = list(zip(*batch))
        batch_sen_pair = list(zip(zipped[0], zipped[1]))
        batch_id = zipped[2]

        max_length = min(max([len(p[0]) + len(p[1]) for p in batch_sen_pair]), 512)
        batch_input_ids = []
        batch_token_type_ids = []
        for sen_pair in batch_sen_pair:
            out = tokenizer.encode_plus(*sen_pair, max_length = max_length, pad_to_max_length = True)
            batch_input_ids.append(out['input_ids'])
            batch_token_type_ids.append(out['token_type_ids'])

        return torch.tensor(batch_input_ids), torch.tensor(batch_token_type_ids), batch_id
    return callback

class TestDataset(data.Dataset):
    def __init__(self, tokenizer):
        self.array = self._parse_data()
        self.tokenizer = tokenizer
        print('finish parsing data')
        
    def _parse_data(self):
        array = []
        senA, senB, _id = test_parse_xlsx(join('data', 'TestingData_fill.xlsx'))
        array += list(zip(senA, senB, _id))
        return array

    def __getitem__(self, i):
        item = self.array[i]
        return item
    
    def __len__(self):
        return len(self.array)
