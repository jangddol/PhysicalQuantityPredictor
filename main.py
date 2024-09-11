import numpy as np
import torch

import matplotlib.pyplot as plt

TIME_LENGTH = 1000
QUANTITY_NUM = 3
SUSCEPT_LENGTH = 30

def make_diff_seq_torch(data_seq):
    # Compute the difference along the second dimension
    diff_seq = torch.diff(data_seq, dim=1)
    
    # Append a column of zeros to match the original shape
    zero_column = torch.zeros(QUANTITY_NUM, 1)
    diff_seq = torch.cat((diff_seq, zero_column), dim=1)
    
    return diff_seq

def make_data_stride_torch(data_seq):
    # Create an index tensor for the strides
    indices = torch.arange(TIME_LENGTH).unsqueeze(1) - torch.arange(SUSCEPT_LENGTH).unsqueeze(0)
    
    # Use advanced indexing to create the data_stride tensor
    data_stride = data_seq[:, indices]

    # mask is 1, then replace with 0
    data_stride[:, indices < 0] = 0
    
    return data_stride

def make_D_tensor(diff_seq, data_stride):
    # diff_seq shape : (QUANTITY_NUM, TIME_LENGTH), index : ik
    # data_stride shape : (QUANTITY_NUM, TIME_LENGTH, SUSCEPT_LENGTH), index : jkl
    # First, we need to cut the k range as [SUSCEPT_LENGTH-1:TIME_LENGTH]
    new_diff_seq = diff_seq[:, SUSCEPT_LENGTH-1:]
    new_data_stride = data_stride[:, SUSCEPT_LENGTH-1:, :]
    D_tensor = torch.einsum('ik,jkl->ijl', new_diff_seq, new_data_stride)
    return D_tensor

def make_Y_tensor(data_stride):
    # data_stride shape : (QUANTITY_NUM, TIME_LENGTH, SUSCEPT_LENGTH), index : jkl
    # copied_strid shape : (QUANTITY_NUM, TIME_LENGTH, SUSCEPT_LENGTH), index : pkq
    # Y_tensor shape : (QUANTITY_NUM, SUSCEPT_LENGTH, QUANTITY_NUM, SUSCEPT_LENGTH), index : pqjl
    # First, we need to cut the k range as [SUSCEPT_LENGHT-1:TIME_LENGTH]
    new_data_stride = data_stride[:, :TIME_LENGTH-SUSCEPT_LENGTH+1:, ]
    Y_tensor = torch.einsum('jkl,pkq->pqjl', new_data_stride, new_data_stride)
    return Y_tensor

def make_susceptibility_tensor(D_tensor, Y_tensor):
    # D_tensor shape : (QUANTITY_NUM, QUANTITY_NUM, SUSCEPT_LENGTH), index : ipq
    # Y_tensor shape : (QUANTITY_NUM, SUSCEPT_LENGTH, QUANTITY_NUM, SUSCEPT_LENGTH), index : pqjl
    # susceptibility shape : (QUANTITY_NUM, QUANTITY_NUM, SUSCEPT_LENGTH), index : ijl
    # My linear equation is D_tensor = susceptibility * Y_tensor
    # So, susceptibility = D_tensor * Y_tensor^-1

    Y_tensor_reshaped = Y_tensor.view(QUANTITY_NUM * SUSCEPT_LENGTH, QUANTITY_NUM * SUSCEPT_LENGTH)
    # Compute the inverse of the reshaped tensor
    Y_tensor_inv_reshaped = torch.inverse(Y_tensor_reshaped)
    # Reshape back to the original 4D shape
    Y_tensor_inv = Y_tensor_inv_reshaped.view(QUANTITY_NUM, SUSCEPT_LENGTH, QUANTITY_NUM, SUSCEPT_LENGTH)
    
    susceptibility = torch.einsum('ipq,pqjl->ijl', D_tensor, Y_tensor_inv)
    return susceptibility

def plot_susceptibility(susceptibility):
    # susceptibility shape : (QUANTITY_NUM, QUANTITY_NUM, SUSCEPT_LENGTH), index : ijl
    # for each i, j, plot the susceptibility with different subplot

    fig, axs = plt.subplots(QUANTITY_NUM, QUANTITY_NUM, figsize=(10, 10))
    for i in range(QUANTITY_NUM):
        for j in range(QUANTITY_NUM):
            axs[i, j].plot(susceptibility[i, j, :].detach().numpy())
            axs[i, j].set_title(f'({i}, {j})')
            axs[i, j].grid(True)
    plt.show()


if __name__ == '__main__':
    # make example data
    row_1 = [np.cos(2*np.pi*0.001*i) for i in range(TIME_LENGTH)]
    row_2 = [np.sin(2*np.pi*0.002*i + 1) for i in range(TIME_LENGTH)]
    row_3 = [np.cos(2*np.pi*0.003*i + 3) for i in range(TIME_LENGTH)]

    data_seq = torch.tensor(np.random.rand(QUANTITY_NUM, TIME_LENGTH))
    diff_seq = make_diff_seq_torch(data_seq)
    data_stride = make_data_stride_torch(data_seq)
    D_tensor = make_D_tensor(diff_seq, data_stride)
    Y_tensor = make_Y_tensor(data_stride)
    susceptibility = make_susceptibility_tensor(D_tensor, Y_tensor)

    plot_susceptibility(susceptibility)