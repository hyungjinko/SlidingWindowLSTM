# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:31:31 2017

@author: khz1234
"""
import tensorflow as tf
import time
import numpy as np
import util

class LSTM:
    def __init__(self, par):
        self.par = par
        self.default_par = {'data_dim':18, 'hidden_dim':10, 'output_dim':1, 'learning_rate':0.01, 'activation':tf.tanh}
        self.data = util.data_loading(self.par['path']) # data loading
        self.scaled_data = util.MinMaxScaler(self.data) # data scailing
        self.root_data, self.testY = util.data_processing1(self.scaled_data, self.par['seq_length']) # data_processing1()
        self.result_predict = []
        self.result_true = []
        self.result_time = []
        
    def LSTM_train_and_infer(self, i):
        with tf.device('/gpu:0'):
            # input place holders
            X = tf.placeholder(tf.float32, [None, self.par['sub_seq_length'], self.default_par['data_dim']])
            Y = tf.placeholder(tf.float32, [None, 1])

            # build a LSTM network
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.default_par['hidden_dim'], state_is_tuple=True, activation=self.default_par['activation'])
            with tf.variable_scope('{}'.format(i)):
                outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
            Y_pred = tf.contrib.layers.fully_connected(
                outputs[:, -1], self.default_par['output_dim'], activation_fn=None)  # We use the last cell's output
    
            # cost/loss
            loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
            # optimizer
            optimizer = tf.train.AdamOptimizer(self.default_par['learning_rate'])
            train = optimizer.minimize(loss)
    
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
        
            # Training step
            for j in range(self.par['iterations']):
                _, step_loss = sess.run([train, loss], feed_dict={
                                        X: self.trainX, Y: self.trainY})
                if j % 100 == 0:
                    print("[step: {}] loss: {}".format(j, step_loss))

            # Test step
            test_predict = sess.run(Y_pred, feed_dict={X: self.testX})
            
        return test_predict
    
    def LSTM_result(self):
        for i in range(len(self.root_data)):
            start = time.time()
            
            # data_processing2()
            xy = self.root_data[i]
            x = xy
            y = xy[:, [-1]]  # output data
            dataX, dataY = util.data_processing2(x, y, self.par['sub_seq_length'])
            
            # data_processing3()
            self.trainX, self.testX, self.trainY, self.testY = util.data_processing3(dataX, dataY)
            
            # LSTM training and test
            _result_predict = self.LSTM_train_and_infer(i)
            _result_true = self.testY
            print("successfully train {}th/{}th sub RNN model".format(i, len(self.root_data) - self.par['seq_length']))
            self.result_predict.append(_result_predict)
            self.result_true.append(_result_true)
            
            end = time.time() - start
            self.result_time.append(end)
        
        self.result_predict = np.array(self.result_predict, dtype = np.float32)
        self.result_true = np.array(self.result_true, dtype = np.float32)
        self.result_time = np.array(self.result_time, dtype = np.float32)
    
        original_price_data = self.data[:,-1]
        self.result_predict = util.ReverseScaler(self.result_predict, original_price_data)
        self.result_true = util.ReverseScaler(self.result_true, original_price_data)
    
        plot_name = str(self.par['seq_length'])+'_'+str(self.par['sub_seq_length'])+'_'+str(self.par['iteration'])
        np.savetxt(self.par['save_path']+'output_predict_{}.csv'.format(plot_name),self.result_predict)
        np.savetxt(self.par['save_path']+'output_true_{}.csv'.format(plot_name),self.result_true)
        np.savetxt(self.par['save_path']+'output_time_{}.csv'.format(plot_name),self.result_time)
        
        print("successfully save the result!")

        
    
