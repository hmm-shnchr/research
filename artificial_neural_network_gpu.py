from multilayer_extend_gpu import MultiLayerNetExtend
from reshape_merger_tree import ReshapeMergerTree
from optimizer_gpu import set_optimizer
import matplotlib.pyplot as plt
import numpy as np
import cupy
import copy as cp
import os, sys, shutil


class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func):
        self.input_size = input_size
        self.hidden, self.act_func, self.weight_init = hidden, act_func, weight_init
        self.batch_norm = batch_norm
        self.output_size = output_size
        self.loss_func = loss_func
        self.network = MultiLayerNetExtend(input_size*2, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func)
        self.loss_val = {}
        self.train_acc, self.test_acc = {}, {}
        
    def learning(self, train, test, opt, lr, batchsize_denominator, epoch, m_list):
        ##Initialize the self-variables.
        for m_key in m_list:
            self.loss_val[m_key] = []
            self.train_acc[m_key], self.test_acc[m_key] = [], []
        ##Make input/output dataset.
        RMT_train, RMT_test = {}, {}
        train_input, train_output = {}, {}
        test_input, test_output = {}, {}
        train_input_, train_output_ = None, None
        test_input_, test_output_ = None, None
        for m_key in m_list:
            RMT_train[m_key] = ReshapeMergerTree()
            RMT_test[m_key] = ReshapeMergerTree()
            train_input[m_key], train_output[m_key] = RMT_train[m_key].make_dataset(train[m_key], self.input_size, self.output_size)
            test_input[m_key], test_output[m_key] = RMT_test[m_key].make_dataset(test[m_key], self.input_size, self.output_size)
            #train_input[m_key], train_output[m_key] = cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key])
            #test_input[m_key], test_output[m_key] = cupy.asarray(test_input[m_key]), cupy.asarray(test_output[m_key])
            if train_input_ is None:
                train_input_, train_output_ = cp.deepcopy(train_input[m_key]), cp.deepcopy(train_output[m_key])
                test_input_, test_output_ = cp.deepcopy(test_input[m_key]), cp.deepcopy(test_output[m_key])
            else:
                #train_input_, train_output_ = cupy.concatenate([train_input_, train_input[m_key]], axis = 0), cupy.concatenate([train_output_, train_output[m_key]], axis = 0)
                #test_input_, test_output_ = cupy.concatenate([test_input_, test_input[m_key]], axis = 0), cupy.concatenate([test_output_, test_output[m_key]], axis = 0)
                train_input_, train_output_ = np.concatenate([train_input_, train_input[m_key]], axis = 0), np.concatenate([train_output_, train_output[m_key]], axis = 0)
                test_input_, test_output_ = np.concatenate([test_input_, test_input[m_key]], axis = 0), np.concatenate([test_output_, test_output[m_key]], axis = 0)
        print("Make train/test dataset.")
        ##Define the optimizer.
        learning_rate = float(lr)
        optimizer = set_optimizer(opt, learning_rate)
        ##Define the number of iterations.
        rowsize_train = train_input_.shape[0]
        batch_mask_arange = np.arange(rowsize_train)
        batch_size = int(rowsize_train/batchsize_denominator)
        iter_per_epoch = int(rowsize_train/batch_size)
        iter_num = iter_per_epoch * epoch
        ##Start learning.
        for i in range(iter_num):
            ##Make a mini batch.
            batch_mask = np.random.choice(batch_mask_arange, batch_size)
            #batch_input, batch_output = train_input_[batch_mask, :], train_output_[batch_mask, :]
            batch_input, batch_output = cupy.asarray(train_input_[batch_mask, :]), cupy.asarray(train_output_[batch_mask, :])
            ##Update the self.network.params with grads.
            grads = self.network.gradient(batch_input, batch_output, is_training = True)
            params_network = self.network.params
            optimizer.update(params_network, grads, i)
            ##When the iteration i reaches a multiple of iter_per_epoch,
            ##Save loss_values, train/test_accuracy_value of the self.network to self.loss_val, self.train_acc, self.test_acc.
            if i % iter_per_epoch == 0:
                for m_key in m_list:
                    #loss_val = self.network.loss(train_input[m_key], train_output[m_key], is_training = False)
                    loss_val = self.network.loss(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                    self.loss_val[m_key].append(loss_val)
                    #train_acc = self.network.accuracy(train_input[m_key], train_output[m_key], is_training = False)
                    train_acc = self.network.accuracy(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                    self.train_acc[m_key].append(train_acc)
                    #test_acc = self.network.accuracy(test_input[m_key], test_output[m_key], is_training = False)
                    test_acc = self.network.accuracy(cupy.asarray(test_input[m_key]), cupy.asarray(test_output[m_key]), is_training = False)
                    self.test_acc[m_key].append(test_acc)
                    
    def predict(self, dataset, RMT_flag = True):
        m_list = dataset.keys()
        prediction = {}
        if RMT_flag:
            RMT = {}
            data_input = {}
            for m_key in m_list:
                RMT[m_key] = ReshapeMergerTree()
                data_input[m_key], _ = RMT[m_key].make_dataset(dataset[m_key], self.input_size, self.output_size)
                data_input[m_key] = cupy.asarray(data_input[m_key])
        else:
            data_input = dataset
        for m_key in m_list:
            prediction[m_key] = self.network.predict(data_input[m_key], is_training = False)
            prediction[m_key] = cupy.asnumpy(prediction[m_key])
        if RMT_flag:
            for m_key in m_list:
                prediction[m_key] = RMT[m_key].restore_mergertree(prediction[m_key])
        return prediction
    
    def plot_figures(self, save_dir, save_fig_type):
        m_list = self.loss_val.keys()
        fontsize = 26
        labelsize = 15
        length_major = 20
        length_minor = 15
        linewidth = 2.5
        for m_key in m_list:
            ##Plot and save the self.loss_val.
            epochs = np.arange(len(self.loss_val[m_key]))
            fig = plt.figure(figsize = (8, 5))
            ax_loss = fig.add_subplot(111)
            ax_loss.plot(epochs, self.loss_val[m_key], label = "Loss Function", color = "red", linewidth = linewidth)
            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epoch", fontsize = fontsize)
            ax_loss.set_ylabel("Loss Function", fontsize = fontsize)
            ax_loss.legend(loc = "best", fontsize = int(fontsize * 0.6))
            ax_loss.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
            ax_loss.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
            plt.title("Loss Function({})".format(m_key[11:16]), fontsize = fontsize)
            plt.tight_layout()
            plt.savefig("{}fig_loss_{}{}".format(save_dir, m_key[11:16], save_fig_type))
            np.savetxt("{}data_loss_{}.csv".format(save_dir, m_key[11:16]), self.loss_val[m_key], delimiter = ",")
            ##Plot and save the self.train/test_acc.
            epochs = np.arange(len(self.train_acc[m_key]))
            fig = plt.figure(figsize = (8, 5))
            ax_acc = fig.add_subplot(111)
            ax_acc.plot(epochs, self.train_acc[m_key], label = "Training", color = "orange", linewidth = linewidth)
            ax_acc.plot(epochs, self.test_acc[m_key], label = "Testing", color = "blue", linewidth = linewidth)
            ax_acc.set_yscale("log")
            ax_acc.set_xlabel("Epoch", fontsize = fontsize)
            ax_acc.set_ylabel("Accuracy", fontsize = fontsize)
            ax_acc.legend(loc = "best", fontsize = int(fontsize * 0.6))
            ax_acc.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
            ax_acc.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
            plt.title("Training and Testing Accuracy({})".format(m_key[11:16]), fontsize = fontsize)
            plt.tight_layout()
            plt.savefig("{}fig_acc_{}{}".format(save_dir, m_key[11:16], save_fig_type))
            np.savetxt("{}data_acc_train_{}.csv".format(save_dir, m_key[11:16]), self.train_acc[m_key], delimiter = ",")
            np.savetxt("{}data_acc_test_{}.csv".format(save_dir, m_key[11:16]), self.test_acc[m_key], delimiter = ",")
