#!/usr/bin/env python

# import the required packages here
from data import train_data_prepare
from train import train
from test import test, test_data_prepare

def run(train_file, valid_file, test_file, output_file):
    '''The function to run your ML algorithm on given datasets, generate the output and save them into the provided file path

    Parameters
    ----------
    train_file: string
        the path to the training file
        valid_file: string
                the path to the validation file
        test_file: string
                the path to the testing file
    output_file: string
        the path to the output predictions to be saved
    '''

    ## your implementation here

    # read data from input
    train_samples, word2num = train_data_prepare(train_file)
    valid_samples = test_data_prepare(valid_file, word2num, 'valid')

    # your training algorithm
    model = train(train_samples, valid_samples, word2num)

    # your prediction code
    test(test_file, output_file, word2num, model)

    # define other functions here


run('train.tsv', 'valid.tsv', 'test.tsv', 'predictions.txt')