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
    #receive_parts = [np.empty_like(s) for s in partition]

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

    recv = np.concatenate(partition, axis=0)
    return recv

    '''for i in range(size * 2):
        send_data_idx = (my_rank - i) % size
        recv_data_idx = (my_rank - i - 1) % size

        send_request = comm.Isend(partition[send_data_idx], sending_idx)

        comm.Recv(receive_parts[recv_data_idx], recv_idx)

        if i < size - 1:  # Reducing
            partition[recv_data_idx] = op(receive_parts[recv_data_idx],
                                                   partition[recv_data_idx])
        else:  # shifting
            partition[recv_data_idx] = receive_parts[recv_data_idx]

        send_request.Wait()

    recv = np.concatenate(partition, 0)
    #np.copyto(recv, toReturn)
    return recv'''

    '''

    with open("debug.txt", "a") as file:
        file.write("Rank is: " + str(my_rank) + ":    Send is;: " + " And Type: " + str(type(send)))
        np.savetxt(file, send)
        file.write("\n")

    

    with open("debug.txt", "a") as file:
        file.write("Rank " + str(my_rank) + ":   Partition: ")
        np.savetxt(file, partition)
        file.write("\n")



    # first loop with calc
    for i in range(size - 1):
        print("My Rank: ", my_rank)

        send_data_idx = (my_rank - i) % size
        recv_data_idx = (my_rank - i - 1) % size

        data_to_send = partition[send_data_idx]

        print("Send: ", data_to_send)

        comm.Isend(data_to_send, dest=sending_idx, tag=1)

        cur_recv = np.zeros_like(len(partition[recv_data_idx]))
        comm.Recv(cur_recv, recv_idx, 1)

        print("Iter is: ", i, "Recv: ", cur_recv, "Have: ", partition[recv_data_idx])
        partition[recv_data_idx] = op(partition[recv_data_idx], cur_recv)
        print("Result: ", partition[recv_data_idx])

    comm.Barrier()

    print("Midway: ", partition)

    # second loop with just sending
    for i in range(size - 1):

        send_data_idx = (my_rank - i) % size
        recv_data_idx = (my_rank - 1 - i) % size

        comm.Isend(partition[send_data_idx], dest=sending_idx, tag=1)

        cur_recv = np.zeros_like(len(partition[recv_data_idx]))
        comm.Recv(cur_recv, recv_idx, 1)

        partition[recv_data_idx] = cur_recv

        comm.Barrier()

    print("RES: ", partition)

    recv = np.concatenate(partition, axis=0)
    return recv'''

    #raise NotImplementedError("To be implemented")
