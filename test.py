import torch
from transformers import *
import torch.utils.data as data
from util import test_custom_collate, TestDataset
import torch.nn.functional as F
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

dataset = TestDataset(tokenizer)
dataloader = data.DataLoader(dataset = dataset, batch_size = 4, shuffle = False, collate_fn = test_custom_collate(tokenizer))

ans_list = []
id_list = []
with torch.no_grad():
    for (batch_idx, item) in enumerate(dataloader):
        # a batch is a q-a pair
        score = model(input_ids = item[0], token_type_ids = item[1])
        score = F.softmax(score[0], dim = 1)[:, 0]
        ans = torch.argmax(score, dim = 0)
        ans_list.append(ans.item())
        id_list.append(item[2][0])
        print("id: {}, batch_idx: {}/{}, ans: {}".format(item[2][0], batch_idx, len(dataloader), ans))

df = pd.DataFrame({
    'ID': id_list,
    'Ans': ans_list
})

df.to_csv('submission.csv', index = False)
