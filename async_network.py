from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        # TODO: add your code
        workless_batches = self.number_of_batches - (self.number_of_batches // self.num_workers) * self.num_workers
        self.number_of_batches = self.number_of_batches // self.num_workers
        if self.rank % self.num_workers < workless_batches:
            self.number_of_batches += 1

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                # TODO: add your code
                recv_req = []
                for layer in range(self.num_layers):
                    cur_master = layer % self.num_masters

                    b_to_send = nabla_b[layer]
                    w_to_send = nabla_w[layer]

                    self.comm.Isend(b_to_send, dest=cur_master, tag=layer)
                    self.comm.Isend(w_to_send, dest=cur_master, tag=layer + self.num_layers)

                    # recieve new self.weight and self.biases values from masters
                    # TODO: add your code

                    recv_req.append(self.comm.Irecv(self.biases[layer], source=cur_master, tag=layer))
                    recv_req.append(self.comm.Irecv(self.weights[layer], source=cur_master, tag=layer + self.num_layers))

                for req in recv_req:
                    req.Wait()

    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        relevant_layers = [layer for layer in range(self.rank, self.num_layers, self.num_masters)]

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code
                stat = MPI.Status()

                layer_num = 0
                cur_worker = -1
                for layer in relevant_layers:
                    if layer_num == 0:
                        self.comm.Recv(nabla_b[layer_num], source=MPI.ANY_SOURCE, tag=layer, status=stat)
                        cur_worker = stat.source
                        self.comm.Recv(nabla_w[layer_num], source=cur_worker, tag=layer + self.num_layers)

                    else:
                        self.comm.Recv(nabla_b[layer_num], source=cur_worker, tag=layer)
                        self.comm.Recv(nabla_w[layer_num], source=cur_worker, tag=layer + self.num_layers)

                    layer_num += 1

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # TODO: add your code
                send_req = []
                for layer in relevant_layers:
                    send_req.append(self.comm.Isend(self.biases[layer], dest=cur_worker, tag=layer))
                    send_req.append(self.comm.Isend(self.weights[layer], dest=cur_worker, tag=layer + self.num_layers))

                for req in send_req:
                    req.Wait()

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
        if self.rank == 0:
            for layer in range(self.num_layers):
                cur_master = layer % self.num_masters
                if cur_master != 0:
                    self.comm.Recv(self.biases[layer], source=cur_master, tag=layer)
                    self.comm.Recv(self.weights[layer], source=cur_master, tag=layer + self.num_layers)

        else:
            send_req = []
            for layer in relevant_layers:
                send_req.append(self.comm.Isend(self.biases[layer], 0, tag=layer))
                send_req.append(self.comm.Isend(self.weights[layer], 0, tag=layer + self.num_layers))

            for req in send_req:
                req.Wait()
