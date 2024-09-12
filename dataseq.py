import torch
from abc import ABC, abstractmethod


class DataSeq(ABC):
    @abstractmethod
    def __init__(self, data_seq, suscept_length):
        self._quantity_num = None
        self._time_length = None
        self._suscept_length = None
        self._data_seq = None
        self._diff_seq = None
        self._data_stride = None
        self._D_tensor = None
        self._Y_tensor = None
        self._susceptibility_tensor = None
    
    @abstractmethod
    def _is_allowed_data_seq(self):
        pass
    
    @abstractmethod
    def _make_diff_seq(self):
        pass
    
    @abstractmethod
    def _make_data_stride(self):
        pass
    
    @abstractmethod
    def _make_D_tensor(self):
        pass
    
    @abstractmethod
    def _make_Y_tensor(self):
        pass

    @abstractmethod
    def _make_susceptibility_tensor(self):
        pass
    
    @property
    def quantity_num(self):
        return self._quantity_num
    
    @property
    def time_length(self):
        return self._time_length
    
    @property
    def suscept_length(self):
        return self._suscept_length
    
    @property
    def data_seq(self):
        return self._data_seq

    @property
    def diff_seq(self):
        return self._diff_seq
    
    @property
    def data_stride(self):
        return self._data_stride
    
    @property
    def D_tensor(self):
        return self._D_tensor
    
    @property
    def Y_tensor(self):
        return self._Y_tensor
    
    @property
    def susceptibility_tensor(self):
        return self._susceptibility_tensor
    
    @abstractmethod
    def get_loss(self):
        return 0
    
    @abstractmethod
    def change_suscept_length(self, new_suscept_length):
        pass

class SingleDataSeq(DataSeq):
    def __init__(self, data_seq, suscept_length):
        super().__init__(data_seq, suscept_length)
        self._data_seq = data_seq
        self._is_allowed_data_seq()
        self._quantity_num = data_seq.shape[0]
        self._time_length = data_seq.shape[1]
        self._suscept_length = suscept_length
        self._diff_seq = self._make_diff_seq()
        self._data_stride = self._make_data_stride()
        self._D_tensor = self._make_D_tensor()
        self._Y_tensor = self._make_Y_tensor()
        self._susceptibility_tensor = self._make_susceptibility_tensor()
        
    def _is_allowed_data_seq(self):
        if not isinstance(self.data_seq, torch.Tensor):
            raise ValueError("data_seq must be a torch.Tensor")
        if len(self.data_seq.shape) != 2:
            raise ValueError("data_seq must be a 2D tensor")
    
    def _make_diff_seq(self):
        # data_seq shape : (quantity_num, time_length), index : ik
        # Compute the difference along the second dimension
        diff_seq = torch.diff(self.data_seq, dim=1)
        
        # Append a column of zeros to match the original shape
        zero_column = torch.zeros(self.quantity_num, 1)
        return torch.cat((diff_seq, zero_column), dim=1)
    
    def _make_data_stride(self):
        # data_seq shape : (quantity_num, time_length), index : ik
        # Create an index tensor for the strides
        indices = torch.arange(self.time_length).unsqueeze(1) - torch.arange(self.suscept_length).unsqueeze(0)
        # Use advanced indexing to create the data_stride tensor
        data_stride = self.data_seq[:, indices]
        # mask is 1, then replace with 0
        data_stride[:, indices < 0] = 0
        return data_stride
    
    def _make_D_tensor(self):
        # diff_seq shape : (quantity_num, time_length), index : ik
        # data_stride shape : (quantity_num, time_length, suscept_length), index : jkl
        # First, we need to cut the k range as [suscept_length-1:time_length]
        new_diff_seq = self.diff_seq[:, self.suscept_length-1:]
        new_data_stride = self.data_stride[:, self.suscept_length-1:, :]
        D_tensor = torch.einsum('ik,jkl->ijl', new_diff_seq, new_data_stride)
        return D_tensor
    
    def _make_Y_tensor(self):
        # data_stride shape : (quantity_num, time_length, suscept_length), index : jkl
        # copied_stride shape : (quantity_num, time_length, suscept_length), index : pkq
        # Y_tensor shape : (quantity_num, suscept_length, quantity_num, suscept_length), index : pqjl
        # First, we need to cut the k range as [suscept_length-1:time_length]
        new_data_stride = self.data_stride[:, :self.time_length-self.suscept_length+1:, ]
        Y_tensor = torch.einsum('jkl,pkq->pqjl', new_data_stride, new_data_stride)
        return Y_tensor
    
    def _make_susceptibility_tensor(self):
        # D_tensor shape : (quantity_num, quantity_num, suscept_length), index : ipq
        # Y_tensor shape : (quantity_num, suscept_length, quantity_num, suscept_length), index : pqjl
        # susceptibility shape : (quantity_num, quantity_num, suscept_length), index : ijl
        # My linear equation is D_tensor = susceptibility * Y_tensor
        # So, susceptibility = D_tensor * Y_tensor^-1
        Y_tensor_reshaped = self.Y_tensor.view(self.quantity_num * self.suscept_length, self.quantity_num * self.suscept_length)
        # Compute the inverse of the reshaped tensor
        try:
            Y_tensor_inv_reshaped = torch.inverse(Y_tensor_reshaped)
        except TypeError:
            raise TypeError("Y tensor is not invertible. Plz try another way.")
        # Reshape back to the original 4D shape
        Y_tensor_inv = Y_tensor_inv_reshaped.view(self.quantity_num, self.suscept_length, self.quantity_num, self.suscept_length)
        
        susceptibility_tensor = torch.einsum('ipq,pqjl->ijl', self.D_tensor, Y_tensor_inv)
        return susceptibility_tensor
    
    def get_loss(self):
        loss_term1 = torch.einsum('ijl,jkl->ik', self.susceptibility_tensor, self.data_stride)
        loss_term2 = self.diff_seq
        loss = torch.sum((loss_term1 - loss_term2) ** 2)
        return loss
    
    def _guess_one_next_data(self, base_last_index):
        base_data_stride = self.data_seq[:, base_last_index - self.suscept_length + 1: base_last_index + 1]
        guessed_data = torch.einsum('ijl,jkl->ik', self.susceptibility_tensor, base_data_stride)
        return guessed_data

    def guess_next_data(self, base_last_index, guessing_length):
        # base_last_index : the last index of the base data
        # guessing_length : the length of the data to be guessed
        guessed_data = torch.zeros(self.quantity_num, guessing_length)
        for i in range(guessing_length):
            guessed_data[:, i] = self._guess_one_next_data(base_last_index + i)
        return guessed_data
    
    def change_suscept_length(self, new_suscept_length):
        self.__init__(self.data_seq, new_suscept_length)

class MultipleDataSeq(DataSeq):
    def __init__(self, data_seq, suscept_length):
        super().__init__(data_seq, suscept_length)
        self._data_seq = data_seq
        self._is_allowed_data_seq()
        self._quantity_num = data_seq[0].quantity_num
        self._time_length = [single_data_seq.time_length for single_data_seq in data_seq]
        self._suscept_length = suscept_length
        self._diff_seq = self._make_diff_seq()
        self._data_stride = self._make_data_stride()
        self._D_tensor = self._make_D_tensor()
        self._Y_tensor = self._make_Y_tensor()
        self._susceptibility_tensor = self._make_susceptibility_tensor()
        
    def _is_allowed_data_seq(self):
        if not isinstance(self.data_seq, list):
            raise ValueError("data_seq must be a list of SingleDataSeq")
        for i, data_seq in enumerate(self.data_seq):
            if i != 0:
                if self.data_seq[0].quantity_num != data_seq.quantity_num:
                    raise ValueError("quantity_num must be the same")
    
    def _make_diff_seq(self):
        return [data_seq.diff_seq for data_seq in self.data_seq]
    
    def _make_data_stride(self):
        return [data_seq.data_stride for data_seq in self.data_seq]
    
    def _make_D_tensor(self):
        D_tensor_list = [data_seq.D_tensor for data_seq in self.data_seq]
        return torch.stack(D_tensor_list).sum(dim=0)
    
    def _make_Y_tensor(self):
        Y_tensor_list = [data_seq.Y_tensor for data_seq in self.data_seq]
        return torch.stack(Y_tensor_list).sum(dim=0)
    
    def _make_susceptibility_tensor(self):
        # D_tensor shape : (quantity_num, quantity_num, suscept_length), index : ipq
        # Y_tensor shape : (quantity_num, suscept_length, quantity_num, suscept_length), index : pqjl
        # susceptibility shape : (quantity_num, quantity_num, suscept_length), index : ijl
        # My linear equation is D_tensor = susceptibility * Y_tensor
        # So, susceptibility = D_tensor * Y_tensor^-1
        Y_tensor_reshaped = self.Y_tensor.view(self.quantity_num * self.suscept_length, self.quantity_num * self.suscept_length)
        # Compute the inverse of the reshaped tensor
        try:
            Y_tensor_inv_reshaped = torch.inverse(Y_tensor_reshaped)
        except TypeError:
            raise TypeError("Y tensor is not invertible. Plz try another way.")
        # Reshape back to the original 4D shape
        Y_tensor_inv = Y_tensor_inv_reshaped.view(self.quantity_num, self.suscept_length, self.quantity_num, self.suscept_length)
        
        susceptibility_tensor = torch.einsum('ipq,pqjl->ijl', self.D_tensor, Y_tensor_inv)
        return susceptibility_tensor

    def get_loss(self):
        loss_term1 = [torch.einsum('ijl,jkl->ik', self.susceptibility_tensor, data_stride) for data_stride in self.data_stride]
        loss_term2 = self.diff_seq
        loss = sum([torch.sum((loss_term1[i] - loss_term2[i]) ** 2) for i in range(len(loss_term1))])
        return loss
    
    def guess_next_data(self, seq_index, base_last_index, guessing_length):
        return self.data_seq[seq_index].guess_next_data(base_last_index, guessing_length)
    
    def change_suscept_length(self, new_suscept_length):
        for data_seq in self.data_seq:
            data_seq.change_suscept_length(new_suscept_length)
        self.__init__(self.data_seq, new_suscept_length)