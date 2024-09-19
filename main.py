import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataseq import MultipleDataSeq, SingleDataSeq

def load_log(file_path):
    # the form of the log file is as follows:
    # 2024-09-06 15:38:49, 0.00, 0.00, 0.00, 0.00, 0.00
    # YYYY-MM-DD HH:MM:SS, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp
    # I want to make a array of the data, including the time and the data
    # I will return time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp arrays
    # It is important that the time string should be casted to a time object
    time_array = []
    Tip_flow = []
    Shield_flow = []
    Bypass_flow = []
    Head_temp = []
    Tip_temp = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split(',')
            time_str = data[0]
            time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            time_array.append(time_obj)
            Tip_flow.append(float(data[1]))
            Shield_flow.append(float(data[2]))
            Bypass_flow.append(float(data[3]))
            Head_temp.append(float(data[4]))
            Tip_temp.append(float(data[5]))
    return np.array(time_array), np.array(Tip_flow), np.array(Shield_flow), np.array(Bypass_flow), np.array(Head_temp), np.array(Tip_temp)

def cut_data_from_to(_from: datetime.datetime, _to: datetime.datetime, time_array: np.ndarray, Tip_flow: np.ndarray, Shield_flow: np.ndarray, Bypass_flow: np.ndarray, Head_temp: np.ndarray, Tip_temp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # I will return the data between _from and _to
    mask = (time_array >= _from) & (time_array <= _to)
    new_time_array = time_array[mask]
    new_Tip_flow = Tip_flow[mask]
    new_Shield_flow = Shield_flow[mask]
    new_Bypass_flow = Bypass_flow[mask]
    new_Head_temp = Head_temp[mask]
    new_Tip_temp = Tip_temp[mask]
    return new_time_array, new_Tip_flow, new_Shield_flow, new_Bypass_flow, new_Head_temp, new_Tip_temp

def plot_susceptibility(susceptibility):
    # susceptibility shape : (QUANTITY_NUM, QUANTITY_NUM, SUSCEPT_LENGTH), index : ijl
    # for each i, j, plot the susceptibility with different subplot
    QUANTITY_NUM, _, _ = susceptibility.shape
    fig, axs = plt.subplots(QUANTITY_NUM, QUANTITY_NUM, figsize=(10, 10))
    for i in range(QUANTITY_NUM):
        for j in range(QUANTITY_NUM):
            axs[i, j].plot(susceptibility[i, j, :].detach().numpy())
            axs[i, j].set_title(f'({i}, {j})')
            axs[i, j].grid(True)
    plt.show()

def show_consistency_for_susceptibility(multiple_data_seq:MultipleDataSeq, max_suscept_length):
    fig, axs = plt.subplots(multiple_data_seq.quantity_num, multiple_data_seq.quantity_num, figsize=(10, 10))
    for l in range(1, max_suscept_length+1):
        multiple_data_seq.change_suscept_length(l)
        for i in range(multiple_data_seq.quantity_num):
            for j in range(multiple_data_seq.quantity_num):
                axs[i, j].plot(multiple_data_seq.susceptibility_tensor[i, j, :].detach().numpy(), color=(l/max_suscept_length, 1 - l/max_suscept_length, 0))
                axs[i, j].set_title(f'({i}, {j})')
                axs[i, j].grid(True)
    plt.show()

def show_loss_by_suscept_length(multiple_data_seq:MultipleDataSeq, max_suscept_length):
    loss_list = []
    for l in range(1, max_suscept_length+1):
        multiple_data_seq.change_suscept_length(l)
        loss_list.append(multiple_data_seq.SSE)
    
    plt.plot(range(1, max_suscept_length+1), loss_list)
    plt.show()

def show_susceptibility_fft(susceptibility):
    # susceptibility shape : (QUANTITY_NUM, QUANTITY_NUM, SUSCEPT_LENGTH), index : ijl
    # for each i, j, plot the susceptibility with different subplot
    QUANTITY_NUM, _, _ = susceptibility.shape
    fig, axs = plt.subplots(QUANTITY_NUM, QUANTITY_NUM, figsize=(10, 10))
    for i in range(QUANTITY_NUM):
        for j in range(QUANTITY_NUM):
            fft = np.fft.fft(susceptibility[i, j, :].detach().numpy())
            axs[i, j].plot(np.abs(fft))
            axs[i, j].set_title(f'({i}, {j})')
            axs[i, j].grid(True)
    plt.show()

if __name__ == '__main__':
    time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp = load_log('log.txt')
    
    time_from = datetime.datetime(2024, 9, 7, 12, 0, 0)
    time_to = datetime.datetime(2024, 9, 8, 3, 0, 0)
    time_array1, Tip_flow1, Shield_flow1, Bypass_flow1, Head_temp1, Tip_temp1 = cut_data_from_to(time_from, time_to, time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp)
    
    time_from = datetime.datetime(2024, 9, 8, 13, 0, 0)
    time_to = datetime.datetime(2024, 9, 9, 3, 0, 0)
    time_array2, Tip_flow2, Shield_flow2, Bypass_flow2, Head_temp2, Tip_temp2 = cut_data_from_to(time_from, time_to, time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp)
    
    time_from = datetime.datetime(2024, 9, 9, 7, 0, 0)
    time_to = datetime.datetime(2024, 9, 9, 19, 40, 0)
    time_array3, Tip_flow3, Shield_flow3, Bypass_flow3, Head_temp3, Tip_temp3 = cut_data_from_to(time_from, time_to, time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp)

    time_from = datetime.datetime(2024, 9, 9, 21, 0, 0)
    time_to = datetime.datetime(2024, 9, 10, 10, 0, 0)
    time_array4, Tip_flow4, Shield_flow4, Bypass_flow4, Head_temp4, Tip_temp4 = cut_data_from_to(time_from, time_to, time_array, Tip_flow, Shield_flow, Bypass_flow, Head_temp, Tip_temp)

    
    SUSCEPT_LENGTH = 110
    data_seq1 = SingleDataSeq(torch.tensor(np.array([Tip_flow1 + Shield_flow1, Head_temp1, Tip_temp1])), SUSCEPT_LENGTH)
    data_seq2 = SingleDataSeq(torch.tensor(np.array([Tip_flow2 + Shield_flow2, Head_temp2, Tip_temp2])), SUSCEPT_LENGTH)
    data_seq3 = SingleDataSeq(torch.tensor(np.array([Tip_flow3 + Shield_flow3, Head_temp3, Tip_temp3])), SUSCEPT_LENGTH)
    # data_seq4 = SingleDataSeq(torch.tensor(np.array([Tip_flow4 + Shield_flow4, Head_temp4, Tip_temp4])), SUSCEPT_LENGTH)

    print(data_seq1.time_length)
    print(data_seq2.time_length)
    print(data_seq3.time_length)
    # print(data_seq4.time_length)

    multi_data_seq = MultipleDataSeq([data_seq1, data_seq2, data_seq3], SUSCEPT_LENGTH)

    plot_susceptibility(multi_data_seq.susceptibility_tensor)
    show_consistency_for_susceptibility(multi_data_seq, SUSCEPT_LENGTH)
    show_loss_by_suscept_length(multi_data_seq, SUSCEPT_LENGTH)

    # multi_data_seq.change_suscept_length(110)
    # susceptibility = multi_data_seq.susceptibility_tensor
    # show_susceptibility_fft(susceptibility)