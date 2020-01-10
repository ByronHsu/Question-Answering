import torch
from transformers import *
from util import Dataset, custom_collate
import torch.utils.data as data

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

optimizer = torch.optim.Adam(model.parameters())

dataset = Dataset(tokenizer)

BATCH_SIZE = 16

dataloader = data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=custom_collate(tokenizer))

n_epoch = 100

model.train()

for e in range(n_epoch):
    for (batch_idx, item) in enumerate(dataloader):
        optimizer.zero_grad()
        loss, score = model(input_ids = item[0], token_type_ids = item[1], next_sentence_label = item[2])
        loss.backward()
        optimizer.step()
        print('EPOCH[{}/{}] BATCH[{}/{}] loss={}'.format(e, n_epoch, batch_idx, len(dataset) / BATCH_SIZE, loss))
        
        if(batch_idx % 1 == 0):
            model.save_pretrained('checkpoints') #-{}'.format(int(e * len(dataset) + batch_idx)))