import torch
from abc import ABC, abstractmethod


class DataSeq(ABC):
    def __init__(self, data_seq, suscept_length):
        self._data_seq = data_seq
        self._quantity_num = data_seq.shape[0] if isinstance(data_seq, torch.Tensor) else data_seq[0].quantity_num
        self._time_length = data_seq.shape[1] if isinstance(data_seq, torch.Tensor) else data_seq[0].time_length
        self._suscept_length = suscept_length
        
        self._is_allowed_data_seq()
        
        self._diff_seq = None
        self._data_stride_reduced_plus_ones = None
        self._C_tensor = None
        self._susc_tensor_reduced_plus_const = None
    
    @abstractmethod
    def _is_allowed_data_seq(self):
        pass
    
    @abstractmethod
    def _make_diff_seq(self):
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
        if self._diff_seq is None:
            self._diff_seq = self._make_diff_seq()
        return self._diff_seq
    
    @property
    @abstractmethod
    def data_stride(self):
        return 0
    
    @property
    @abstractmethod
    def C_tensor(self):
        return 0
    
    @property
    @abstractmethod
    def susceptibility_tensor(self):
        return 0
    
    @property
    @abstractmethod
    def differential_bias_tensor(self):
        return 0
    
    @property
    @abstractmethod
    def projection_tensor(self):
        return 0
    
    @property
    @abstractmethod
    def SSE(self):
        return 0
    
    @property
    @abstractmethod
    def degree_of_freedom(self):
        return 0
    
    @property
    @abstractmethod
    def standard_error_susceptibility(self):
        return 0
    
    @property
    @abstractmethod
    def covariance_matrix(self):
        return 0
    
    @abstractmethod
    def change_suscept_length(self, new_suscept_length):
        pass

class SingleDataSeq(DataSeq):
    def __init__(self, data_seq, suscept_length):
        super().__init__(data_seq, suscept_length)
        
    def _is_allowed_data_seq(self):
        if not isinstance(self.data_seq, torch.Tensor):
            raise ValueError("data_seq must be a torch.Tensor")
        if len(self.data_seq.shape) != 2:
            raise ValueError("data_seq must be a 2D tensor")
        # if self.degree_of_freedom <= 0:
            # raise ValueError("time_length is too short")
    
    def _make_diff_seq(self):
        """
        make Delta tensor from data_seq. See also README.md for more details.

        Returns:
            torch.Tensor: Delta tensor (dim = (quantity_num, time_length - suscept_length))
        """
        diff_seq = torch.diff(self.data_seq, dim=1)[:, self.suscept_length - 1:]
        return diff_seq
    
    @property
    def data_stride(self):
        """
        This function creates a data_stride tensor from the data_seq tensor.
        This function do change as follows:
        X_{j, k-l} -> X_{j, k, l} -> X_{k, j, l} -(reshape)-> X_{k, j*l} -(concat)-> X_{k, j*l+1}

        Returns:
            torch.Tensor : data_stride tensor (dim = (time_length - suscept_length, quantity_num * suscept_length + 1))
        """
        
        if self._data_stride_reduced_plus_ones is not None:
            return self._data_stride_reduced_plus_ones
        indices = torch.arange(self.time_length).unsqueeze(1) - torch.arange(self.suscept_length).unsqueeze(0)
        data_stride = self.data_seq[:, indices]
        data_stride[:, indices < 0] = 0
        data_stride = torch.cat((data_stride.permute(1, 0, 2)
                                 .reshape(self.time_length, self.quantity_num * self.suscept_length),
                                 torch.ones((self.time_length, 1))), dim=1)[self.suscept_length-1:self.time_length-1, :]
        self._data_stride_reduced_plus_ones = data_stride
        return data_stride
    
    @property
    def C_tensor(self):
        """
        This function calculates the C tensor from the data_stride tensor.
        This function do calculation as follows:
        C = (X^T * X)^(-1)

        Returns:
            torch.Tensor : C tensor (dim = (quantity_num * suscept_length + 1, quantity_num * suscept_length + 1))
        """
        if self._C_tensor is not None:
            return self._C_tensor
        C_tensor_inv = torch.einsum('ij,ik->jk', self.data_stride, self.data_stride)
        C_tensor = torch.inverse(C_tensor_inv)
        self._C_tensor = C_tensor
        return C_tensor
    
    def _make_suscept_tensor_reduced_plus_const(self):
        suscept_tensor = (self.C_tensor @ self.data_stride.t() @ self.diff_seq.t()).t()
        return suscept_tensor
    
    @property
    def susceptibility_tensor(self):
        if self._susc_tensor_reduced_plus_const is None:
            self._susc_tensor_reduced_plus_const = self._make_suscept_tensor_reduced_plus_const()
        return self._susc_tensor_reduced_plus_const[:, :-1].reshape(self.quantity_num, self.quantity_num, self.suscept_length)
    
    @property
    def differential_bias_tensor(self):
        if self._susc_tensor_reduced_plus_const is None:
            self._susc_tensor_reduced_plus_const = self._make_suscept_tensor_reduced_plus_const()
        return self._susc_tensor_reduced_plus_const[:, -1].unsqueeze(1)
    
    @property
    def projection_tensor(self):
        """
        This function calculates the projection tensor from the data_stride tensor and the C tensor.
        This function do calculation as follows:
        P = X * C * X^T

        Returns:
            torch.Tensor : projection tensor (dim = (time_length - suscept_length, time_length - suscept_length))
        """
        return self.data_stride @ self.C_tensor @ self.data_stride.t()
    
    @property
    def SSE(self):
        """
        This function calculates the sum of squared errors of the data sequence.
        This function do calculation as follows:
        SSE_{i} = (Y * (1 - P) * Y^T)_{ii}

        Returns:
            torch.Tensor : sum of squared errors (dim = (quantity_num))
        """
        P = self.projection_tensor
        Y = self.diff_seq
        SSE = torch.einsum('ij,ij->i', Y, Y - Y @ P)
        return SSE
    
    @property
    def degree_of_freedom(self):
        """
        This function calculates the degree of freedom of the data sequence.
        Note that the degree of freedom is defined as follows:
        degree_of_freedom = quantity_num * (quantity_num * (time_length - 2 * suscept_length) - 1)
        Here, this is the degree of freedom of the total linear model. However, in fact, this model is independent quantity_num * linear_model.
        Therefore, for each linear model, the degree of freedom is quantity_num * (time_length - 2 * suscept_length) - 1.

        Returns:
            int: degree of freedom
        """
        assert self.time_length >= 2 * self.suscept_length # TODO: chenge this condition to raise, but not here
        assert isinstance(self.time_length, int) # TODO: chenge this condition to raise, but not here
        assert self.suscept_length > 0 # TODO: chenge this condition to raise, but not here
        assert isinstance(self.suscept_length, int) # TODO: chenge this condition to raise, but not here
        assert self.quantity_num > 0 # TODO: chenge this condition to raise, but not here
        assert isinstance(self.quantity_num, int) # TODO: chenge this condition to raise, but not here
        
        return self.quantity_num * (self.quantity_num * (self.time_length - 2 * self.suscept_length) - 1)
    
    @property
    def standard_error_susceptibility(self):
        """
        This function calculates the standard error of the susceptibility tensor.
        This function do calculation as follows:
        SE_{ijl} = sqrt( (SSE_{i} / degree_of_freedom) * C_{j*l j*l} )
        
        Returns:
            torch.Tensor : standard error of the susceptibility tensor (dim = (quantity_num, quantity_num, suscept_length))
        """
        SE = torch.sqrt(self.SSE.unsqueeze(1) / self.degree_of_freedom * torch.diagonal(self.C_tensor)[:-1].reshape(self.quantity_num, self.suscept_length))
        return SE
    
    @property
    def covariance_matrix(self):
        """
        This function calculates the covariance matrix of the susceptibility tensor.
        This function do calculation as follows:
        Cov_{ijlj'l'} = sqrt( (SSE_{i} / degree_of_freedom) * C_{j*l j'*l'} )

        Returns:
            torch.Tensor : covariance matrix of the susceptibility tensor (dim = (quantity_num, quantity_num, suscept_length, quantity_num, suscept_length))
        """
        return torch.sqrt(torch.einsum('i,jklm->ijklm', self.SSE, self.C_tensor[:-1, :-1].reshape(self.quantity_num, self.suscept_length, self.quantity_num, self.suscept_length)))
    
    def _guess_one_next_data(self, base_last_index):
        base_data_stride = self.data_seq[:, base_last_index - self.suscept_length + 1: base_last_index + 1]
        guessed_data = torch.einsum('ijl,jkl->ik', self.susceptibility_tensor, base_data_stride)
        return guessed_data

    def guess_next_data(self, base_last_index, guessing_length):
        guessed_data = torch.zeros(self.quantity_num, guessing_length)
        for i in range(guessing_length):
            guessed_data[:, i] = self._guess_one_next_data(base_last_index + i)
        return guessed_data
    
    def change_suscept_length(self, new_suscept_length):
        self.__init__(self.data_seq, new_suscept_length)

class MultipleDataSeq(DataSeq):
    def __init__(self, data_seq: list[SingleDataSeq], suscept_length):
        super().__init__(data_seq, suscept_length)
        
    def _is_allowed_data_seq(self):
        if not isinstance(self.data_seq, list):
            raise ValueError("data_seq must be a list of SingleDataSeq")
        for i, data_seq in enumerate(self.data_seq):
            if i != 0:
                if self.data_seq[0].quantity_num != data_seq.quantity_num:
                    raise ValueError("quantity_num must be the same")
    
    def _make_diff_seq(self):
        return torch.cat([data_seq.diff_seq for data_seq in self.data_seq], dim=1)
    
    @property
    def data_stride(self):
        data_strides = [data_seq.data_stride for data_seq in self.data_seq]
        concatenated_data_stride = torch.cat(data_strides, dim=0)
        ones_column = torch.ones((concatenated_data_stride.shape[0], 1))
        return torch.cat((concatenated_data_stride, ones_column), dim=1)
    
    @property
    def C_tensor(self): # TODO : always the last column has 0 diagonal. can't inverse the tensor.
        if self._C_tensor is not None:
            return self._C_tensor
        C_tensor_inv = torch.einsum('ij,ik->jk', self.data_stride, self.data_stride)
        C_tensor = torch.inverse(C_tensor_inv)
        self._C_tensor = C_tensor
        return C_tensor
    
    def _make_suscept_tensor_reduced_plus_const(self):
        suscept_tensor = (self.C_tensor @ self.data_stride.t() @ self.diff_seq.t()).t()
        return suscept_tensor
    
    @property
    def susceptibility_tensor(self):
        if self._susc_tensor_reduced_plus_const is None:
            self._susc_tensor_reduced_plus_const = self._make_suscept_tensor_reduced_plus_const()
        return self._susc_tensor_reduced_plus_const[:, :-1].reshape(self.quantity_num, self.quantity_num, self.suscept_length)
    
    @property
    def differential_bias_tensor(self):
        if self._susc_tensor_reduced_plus_const is None:
            self._susc_tensor_reduced_plus_const = self._make_suscept_tensor_reduced_plus_const()
        return self._susc_tensor_reduced_plus_const[:, -1].unsqueeze(1)
    
    @property
    def projection_tensor(self):
        return self.data_stride @ self.C_tensor @ self.data_stride.t()
    
    @property
    def SSE(self):
        P = self.projection_tensor
        Y = self.diff_seq
        SSE = torch.einsum('ij,ij->i', Y, Y - Y @ P)
        return SSE
    
    @property
    def degree_of_freedom(self):
        return 0
    
    @property
    def standard_error_susceptibility(self):
        return 0
    
    @property
    def covariance_matrix(self):
        return 0
    
    def _guess_one_next_data(self, base_last_index):
        base_data_stride = self.data_seq[:, base_last_index - self.suscept_length + 1: base_last_index + 1]
        guessed_data = torch.einsum('ijl,jkl->ik', self.susceptibility_tensor, base_data_stride)
        return guessed_data

    def guess_next_data(self, base_last_index, guessing_length):
        guessed_data = torch.zeros(self.quantity_num, guessing_length)
        for i in range(guessing_length):
            guessed_data[:, i] = self._guess_one_next_data(base_last_index + i)
        return guessed_data
    
    def change_suscept_length(self, new_suscept_length):
        self.__init__(self.data_seq, new_suscept_length)
    