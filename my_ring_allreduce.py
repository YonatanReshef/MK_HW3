import numpy as np


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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

    partition = np.array_split(send, size)

    sending_idx = (my_rank + 1) % size
    recv_idx = (my_rank - 1) % size

    # first loop with calc
    for i in range(size - 1):
        send_data_idx = (my_rank - i) % size
        recv_data_idx = (my_rank - i - 1) % size

        data_to_send = partition[send_data_idx]

        send_request = comm.Isend(data_to_send, sending_idx)

        cur_recv = np.empty_like(partition[recv_data_idx])
        comm.Recv(cur_recv, recv_idx)

        partition[recv_data_idx] = op(partition[recv_data_idx], cur_recv)

        send_request.Wait()


    # second loop with just sending
    for i in range(size - 1, 2 * size):
        send_data_idx = (my_rank - i) % size
        recv_data_idx = (my_rank - 1 - i) % size

        send_request = comm.Isend(partition[send_data_idx], dest=sending_idx)

        cur_recv = np.empty_like(partition[recv_data_idx])
        comm.Recv(cur_recv, recv_idx)

        partition[recv_data_idx] = cur_recv

        send_request.Wait()

    res = np.concatenate(partition, axis=0)
    np.copyto(recv, res)
    return recv

    #raise NotImplementedError("To be implemented")
