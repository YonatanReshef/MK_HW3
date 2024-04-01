import numpy as np

from network import *
from my_ring_allreduce import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for epoch in range(self.epochs):

            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)

            for x, y in mini_batches:
                # doing props
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)

                # summing all ma_nabla_b and ma_nabla_w to nabla_w and nabla_b
                nabla_w = ma_nabla_w
                nabla_b = ma_nabla_b
                # TODO: add your code
                if rank == 0:
                    cur_nabla_w = np.zeros_like(ma_nabla_w)
                    cur_nabla_b = np.zeros_like(ma_nabla_b)
                    for rank in range(size - 1):
                        comm.recv(cur_nabla_w, rank + 1, 11)
                        comm.recv(cur_nabla_b, rank + 1, 22)

                        nabla_w = [np.add(sub_grad_1, sub_grad_2) for sub_grad_1, sub_grad_2 in zip(nabla_w, cur_nabla_w)]
                        nabla_b = [np.add(sub_grad_1, sub_grad_2) for sub_grad_1, sub_grad_2 in zip(nabla_b, cur_nabla_b)]

                else:
                    comm.send(ma_nabla_w, 0, 11)
                    comm.send(ma_nabla_b, 0, 22)

                comm.Bcast(nabla_b, 0)
                comm.Bcast(nabla_w, 0)

                # calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
