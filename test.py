import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import DataSample, dataset_to_variable
import numpy as np

num_to_label = ['pants-fire',
                'false',
                'barely-true',
                'half-true',
                'mostly-true',
                'true']

def find_word(word2num, token):
    if token in word2num:
        return word2num[token]
    else:
        return word2num['<unk>']

def test_data_prepare(test_file, word2num, phase):
    test_input = open(test_file, 'rb')
    test_data = test_input.read().decode('utf-8')
    test_input.close()

    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]

    test_samples = []

    for line in test_data.strip().split('\n'):
        tmp = line.strip().split('\t')
        while len(tmp) != 8:
            tmp.append('')
        if phase == 'test':
            p = DataSample('test', tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6])
        elif phase == 'valid':
            p = DataSample(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7])

        for i in range(len(p.statement)):
            p.statement[i] = find_word(statement_word2num, p.statement[i])
        for i in range(len(p.subject)):
            p.subject[i] = find_word(subject_word2num, p.subject[i])
        p.speaker = find_word(speaker_word2num, p.speaker)
        for i in range(len(p.speaker_pos)):
            p.speaker_pos[i] = find_word(speaker_pos_word2num, p.speaker_pos[i])
        p.state = find_word(state_word2num, p.state)
        p.party = find_word(party_word2num, p.party)
        for i in range(len(p.context)):
            p.context[i] = find_word(context_word2num, p.context[i])

        test_samples.append(p)

    return test_samples

def test(test_file, test_output, word2num, model, use_cuda = False):
    test_samples = test_data_prepare(test_file, word2num, 'test')
    dataset_to_variable(test_samples, use_cuda)
    out = open(test_output, 'w')

    for sample in test_samples:
        prediction = model(sample)
        prediction = int(np.argmax(prediction.data.numpy()))
        out.write(num_to_label[prediction]+'\n')

    out.close()

def valid(valid_samples, word2num, model):
    acc = 0
    for sample in valid_samples:
        prediction = model(sample)
        prediction = int(np.argmax(prediction.data.numpy()))
        if prediction == sample.label:
            acc += 1
    acc /= len(valid_samples)
    print('  Validation Accuracy: '+str(acc))
