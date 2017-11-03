# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:27:43 2017

@author: hyungjinko
"""
import numpy as np
import csv


def numerator(data): # numerator for scaling
    numerator = data - np.min(data, 0)
    
    return numerator

def denominator(data): # denominator for scaling
    denominator = np.max(data, 0) - np.min(data, 0)
    
    return denominator + 1e-7

def MinMaxScaler(data): # scaling function
    _numerator = numerator(data)
    _denominator = denominator(data)
    # noise term prevents the zero division
    
    return _numerator / _denominator

def ReverseScaler(scaled_data, original_data): # reverse of scaling
    result = scaled_data * denominator(original_data) + np.min(original_data, 0)
    
    return result

def data_loading(path): # read csv file
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    data = []
    for line in rdr:
        data.append(line)
    f.close()
    data = data[1:]
    data = np.array(data, dtype=np.float32)
    
    return data

def data_processing1(data, seq_length): # root data and true y
    root_data = []
    testY = []
    for j in range(0, len(data) - seq_length):
        _data = data[j:j+seq_length]
        root_data.append(_data)
        _y = []
        _y.append(data[j+seq_length][-1])
        _testY = np.array(_y, dtype = np.float32)
        testY.append(_testY)
        
    return root_data, testY 

def data_processing2(x, y, sub_seq_length):
    # build a dataset
    sub_dataX = []
    sub_dataY = []
    for i in range(0, len(y) - sub_seq_length):
        _x = x[i:i + sub_seq_length]
        _y = y[i + sub_seq_length]  # Next close price
        sub_dataX.append(_x)
        sub_dataY.append(_y)

    return sub_dataX, sub_dataY

def data_processing3(dataX, dataY):
    # train/test split
    train_size = int(len(dataY)-1)
    #test_size = len(dataY) - train_size +1
    trainX = np.array(dataX[0:train_size])
    testX = np.array(dataX[train_size:len(dataX)])
    trainY = np.array(dataY[0:train_size])
    testY = np.array(dataY[train_size:len(dataY)])

    return trainX, testX, trainY, testY