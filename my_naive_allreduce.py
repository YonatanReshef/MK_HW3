import numpy as np


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """

    my_rank = comm.Get_rank()
    size = comm.Get_size()

    recv = send
    cur_recv = np.zeros_like(recv)

    #send to all
    for rank in range(size):
        if rank != my_rank:
            comm.Send(send, dest=rank, tag=1)

    #recv from all and calc result
    for rank in range(size):
        if rank != my_rank:
            comm.Recv(cur_recv, rank, 1)

            for i in range(len(recv)):
                recv[i] = op(recv[i], cur_recv[i])





    #raise NotImplementedError("To be implemented")
