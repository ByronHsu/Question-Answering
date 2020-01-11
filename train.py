import torch
from transformers import *
from util import Dataset, custom_collate
import torch.utils.data as data
import argparse
from tensorboardX import SummaryWriter
import time
import signal
import sys
import os
from os.path import join


def handler(signum, frame):
    '''
    close tensorboad when press ctrl + c
    '''
    print('wait!')
    writer.close()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

args = argparse.ArgumentParser(description='NLP HW4')
args.add_argument('--name', default='baseline', type=str, help='The name of this task')
args.add_argument('--n_epoch', default=100, type=int, help='Number of epochs')
args.add_argument('--batch_size', default=8, type=int, help='Batch size')
args.add_argument('--save_rate', default=50, type=int, help='The frequency to save checkpoint')
args.add_argument('--plot_rate', default=10, type=int, help='The frequency to plot loss')
args.add_argument('--checkpoint', default=None, type=str, help='Resume from the given checkpoint.')

opt = args.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

if(opt.checkpoint != None):
    print('Load model from checkpoints')
    model = BertForNextSentencePrediction.from_pretrained(join('checkpoints', opt.checkpoint))
else:
    model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

dataset = Dataset(tokenizer)
dataloader = data.DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle = True, collate_fn=custom_collate(tokenizer))

task_name = '{}-{}'.format(opt.name, time.strftime("%Y-%m-%d-%H-%M-%S"))
writer = SummaryWriter('runs/{}'.format(task_name))

if not os.path.exists(join('checkpoints', task_name)):
    os.mkdir(join('checkpoints', task_name))

model.to('cuda')
model.train()

acc_loss = [] # accumulated loss, refreshed when it is plotted

for e in range(opt.n_epoch):
    for (batch_idx, item) in enumerate(dataloader):
        optimizer.zero_grad()
        loss, score = model(input_ids = item[0].to('cuda'), token_type_ids = item[1].to('cuda'), next_sentence_label = item[2].to('cuda'))
        print(item[2], score)
        loss.backward()
        optimizer.step()
        print('EPOCH[{}/{}] BATCH[{}/{}] loss={:.3f}'.format(e, opt.n_epoch, batch_idx, len(dataloader), loss))
        
        iter_count = e * len(dataloader) + batch_idx # how many batch iteration
        acc_loss.append(loss.item())

        if(iter_count % opt.save_rate == 0):
            model.save_pretrained(join('checkpoints', task_name))

        if(iter_count > 0 and iter_count % opt.plot_rate == 0):
            avg_loss = sum(acc_loss) / len(acc_loss)
            writer.add_scalar('loss', avg_loss)
            avg_loss = []