import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from model import Net
from test import valid
from data import dataset_to_variable

def train(train_samples,
          valid_samples,
          word2num,
          lr = 0.001,
          epoch = 5,
          use_cuda = False):

    print('Training...')

    # Prepare training data
    print('  Preparing training data...')
    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]

    train_data = train_samples
    dataset_to_variable(train_data, use_cuda)
    valid_data = valid_samples
    dataset_to_variable(valid_data, use_cuda)

    # Construct model instance
    print('  Constructing network model...')
    model = Net(len(statement_word2num),
                len(subject_word2num),
                len(speaker_word2num),
                len(speaker_pos_word2num),
                len(state_word2num),
                len(party_word2num),
                len(context_word2num))
    if use_cuda: model.cuda()

    # Start training
    print('  Start training')

    optimizer = optim.Adam(model.parameters(), lr = lr)
    model.train()

    step = 0
    display_interval = 2000

    for epoch_ in range(epoch):
        print('  ==> Epoch '+str(epoch_)+' started.')
        random.shuffle(train_data)
        total_loss = 0
        for sample in train_data:

            optimizer.zero_grad()

            prediction = model(sample)
            label = Variable(torch.LongTensor([sample.label]))
            loss = F.cross_entropy(prediction, label)
            loss.backward()
            optimizer.step()

            step += 1
            if step % display_interval == 0:
                print('    ==> Iter: '+str(step)+' Loss: '+str(loss))

            total_loss += loss.data.numpy()

        print('  ==> Epoch '+str(epoch_)+' finished. Avg Loss: '+str(total_loss/len(train_data)))

        valid(valid_data, word2num, model)

    return model