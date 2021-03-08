from multilayer_extend import MultiLayerNetExtend
from reshape_merger_tree import ReshapeMergerTree
from optimizer import set_optimizer
import matplotlib.pyplot as plt
import numpy as np
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
        for m_key in m_list:
            RMT_train[m_key] = ReshapeMergerTree()
            RMT_test[m_key] = ReshapeMergerTree()
            train_input[m_key], train_output[m_key] = RMT_train[m_key].make_dataset(train[m_key], self.input_size, self.output_size)
            test_input[m_key], test_output[m_key] = RMT_test[m_key].make_dataset(test[m_key], self.input_size, self.output_size)
        ##Define the optimizer.
        learning_rate = float(lr)
        optimizer = set_optimizer(opt, learning_rate)
        ##Define the number of iterations.
        rowsize_train = 0
        for m_key in m_list:
            rowsize_train += train_input[m_key].shape[0]
        batch_size = int(rowsize_train/batchsize_denominator)
        iter_per_epoch = int(rowsize_train/batch_size)
        iter_num = iter_per_epoch * epoch
        ##Start learning.
        for i in range(iter_num):
            ##Make a mini batch.
            batch_input, batch_output = None, None
            for m_key in m_list:
                rowsize = train_input[m_key].shape[0]
                batch_mask = np.random.choice(np.arange(rowsize), int(rowsize/batchsize_denominator))
                if batch_input is None and batch_output is None:
                    batch_input = train_input[m_key][batch_mask, :]
                    batch_output = train_output[m_key][batch_mask, :]
                else:
                    batch_input = np.concatenate([batch_input, train_input[m_key][batch_mask, :]], axis = 0)
                    batch_output = np.concatenate([batch_output, train_output[m_key][batch_mask, :]], axis = 0)
            ##Update the self.network.params with grads.
            grads = self.network.gradient(batch_input, batch_output, is_training = True)
            params_network = self.network.params
            optimizer.update(params_network, grads, i)
            ##When the iteration i reaches a multiple of iter_per_epoch,
            ##Save loss_values, train/test_accuracy_value of the self.network to self.loss_val, self.train_acc, self.test_acc.
            if i % iter_per_epoch == 0:
                for m_key in m_list:
                    loss_val = self.network.loss(train_input[m_key], train_output[m_key], is_training = False)
                    self.loss_val[m_key].append(loss_val)
                    train_acc = self.network.accuracy(train_input[m_key], train_output[m_key], is_training = False)
                    self.train_acc[m_key].append(train_acc)
                    test_acc = self.network.accuracy(test_input[m_key], test_output[m_key], is_training = False)
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
        else:
            data_input = dataset
        for m_key in m_list:
            prediction[m_key] = self.network.predict(data_input[m_key], is_training = False)
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
