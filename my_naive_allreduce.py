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
    
    #doesnt work
    """
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    cur_recv = np.zeros_like(recv)
    res = np.copy(send)

    #send to all
    for rank in range(size):
        if rank != comm.Get_rank():
            comm.Send(send, dest=rank, tag=1)

    #recv from all and calc result
    for rank in range(size):
        if rank != comm.Get_rank():
            comm.Recv(cur_recv, rank, 1)

            res = op(cur_recv, res)
            
    np.copyto(recv, res)
    return recv
    """

    # working one
    
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    cur_recv = np.zeros_like(send)
    res = np.copy(send)
    
    for rank in range(size):
        if rank == my_rank:
            comm.Bcast(send, rank)
        
        else:
            comm.Bcast(cur_recv, rank)
            res = op(res, cur_recv)
    
    np.copyto(recv, res) # important copyto and not copy!!!
    return recv
    