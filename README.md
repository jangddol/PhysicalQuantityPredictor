# 1. Introduction

This project is a linear regression tool to get pseudo-susceptibility. This pseudo-susceptibility $\chi$ is defined as following.

$$\Delta t \cdot \frac{d}{dt}y^{(i)}(t) = \int_{0}^{\infty} dt' \sum_{j=1}^N \;\chi^{(i)}_{(j)}(t')\,y^{(j)}(t - t')$$

Here, $\Delta t$ is a interval time of discrete time sequence.

More strictly, in discrete time, pseudo-susceptibility $\chi$ is defined as

$$\Delta^{(i)}(t_k) = \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i)}_{(j)} (l \cdot \Delta t) y^{(j)} (t _{k-l}) $$

where,

$$
\begin{aligned}
t_k &= t_0 + k \cdot \Delta t
\\
\Delta^{(i)}(t_k) &= y^{(i)}(t _{k+1}) - y^{(i)}(t_k)
\\
N &= \text{the number of the physical quantities}
\\
y^{(i)}(t_k) &= \text{time sequences of i-th physical quantity}
\\
l &= \text{length to calculate pseudo-susceptibility}
\end{aligned}
$$

# 2. How to Use
## 1. SingleDataSeq

First, make arrays for each physical quantities. Here, I prepare 3 quantities : total_flow, coldtip_temp, head_temp. Note that these quantities array should have same length (this is the time_length).
``` python
suscept_length = 10
data_seq = SingleDataSeq(torch.tensor([total_flow, coldtip_temp, head_temp], suscept_length)
```

By definition the SingleDataSeq object, all calculation is finished.
``` python
Y_tensor = data_seq.Y_tensor
D_tensor = data_seq.D_tensor
susceptibility_tensor = data_seq.susceptibility_tensor
```

If you want to change suscept_length, use "self.change_suscept_length(new_suscept_length)". In fact, this method is just same with re-definition of SingleDataSeq.

Also, you can make the data for future by using "self.guess_next_data(base_last_index, guess_length)".
``` python
base_last_index = data_seq.time_length - 1
guess_length = 100
future_tensor = data_seq.guess_next_data(base_last_index, guess_length)
concat_tensor = torch.concat(data_seq.data_seq, future_tensor, dim=1)
```

## 2. MultipleDataSeq

If you have multiple data sequences with variable time_length's, then you can use MultipleDataSeq.
``` python
data_seq1 = SingleDataSeq(torch.tensor([total_flow1, coldtip_temp1, head_temp1], suscept_length)
...
data_seq5 = SingleDataSeq(torch.tensor([total_flow5, coldtip_temp5, head_temp5], suscept_length)

multi_data_seq = MultipleDataSeq([data_seq1, ... , data_seq5], suscept_length)

Y_tensor = multi_data_seq.Y_tensor
D_tensor = multi_data_seq.D_tensor
susceptibility_tensor = multi_data_seq.susceptibility_tensor
```

# 3. Theoritical Equations
## 1. Solving Pseudo-Susceptibility
Define the loss as following

$$\mathscr{L} = \sum _{i=1}^N \sum _{k=n-1}^{T-1} \left(\Delta^{(i)} _{k} - \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i)} _{jl} y _{j, k-l} \right)^2$$

where,

$$\begin{aligned}
\Delta^{(i)} _k &= \Delta^{(i)}(t_k)
\\
\chi^{(i)} _{jl} &= \chi^{(i)} _{(j)}(l\cdot\Delta t)
\\
y _{j, k-l} &= y^{(j)}(t _{k-l})
\end{aligned}$$

Find minimum by using derivative:

$$\frac{\partial \mathscr{L}}{\partial \chi^{(i')} _{j'l'}} = 2\sum _{k=n-1}^{T-1} \left( \Delta^{(i')} _{k} - \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i')} _{jl} y _{j,k-l} \right) \left( -y _{j', k-l'} \right) = 0$$

Therefore,

$$\sum _{k=n-1}^{T-1} \Delta^{(i')} _k y _{j',k-l'} = \sum _{k=n-1}^{T-1} \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i')} _{jl} y _{j, k-l} y _{j', k-l'}$$

Here, define new tensors: **D-tensor** and **Y_tensor**

$$\begin{aligned}
D^{(i)} _{jl} &\equiv \sum _{k=n-1}^{T-1} \Delta^{(i)} _k y _{j,k-l}
\\
Y _{j'l'jl} &\equiv \sum _{k=n-1}^{T-1} y _{j, k-l} y _{j', k-l'}
\end{aligned}$$

Then, the final linear equation is the following:

$$D^{(i')} _{j'l'} = \sum _{j=1}^N \sum _{l=0}^{n-1} Y _{j'l'jl} \,\chi^{(i')} _{jl}$$

Therefore, the pseudo-susceptibility is

$$\chi^{(i')} _{j'l'} = \sum _{j=1}^N \sum _{l=0}^{n-1} Y^{-1} _{j'l'jl} D^{(i')} _{jl}$$

## 2. w/ Multiple Sequences
If you have multiple time sequences for same system, then you need to change the equations a little bit.

$$\begin{aligned}
\Delta^{(i)}(t^{[a]} _{k_a}) &= \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i)} _{(j)}(l\cdot \Delta t) y^{(j)} (t^{[a]} _{k-l})
\\
\mathscr{L} &= \sum _{a=1}^A \sum _{k_a=n-1}^{T_a-1} \sum _{i=1}^N \left( \Delta^{(i)[a]} _{k} - \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i)} _{jl} y^{[a]} _{j, k-l} \right)^2 
\\
\frac{\partial \mathscr{L}}{\partial \chi^{(i')} _{j'l'}} &= 2 \sum _{a=1}^A \sum _{k_a=n-1}^{T_a-1} \left( \Delta^{(i')[a]} _{k_a} - \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i')} _{jl} y^{[a]} _{j,k-l} \right) \left( -y^{[a]} _{j', k-l'} \right) = 0
\\
\sum _{a=1}^A \sum _{k=n-1}^{T-1} \Delta^{(i')[a]} _{k_a} y^{[a]} _{j',k-l'} &= \sum _{a=1}^A \sum _{k_a=n-1}^{T_a-1} \sum _{j=1}^N \sum _{l=0}^{n-1} \chi^{(i')} _{jl} y^{[a]} _{j, k-l} y^{[a]} _{j', k-l'}
\\
D^{(i)} _{jl} &\equiv \sum _{a=1}^A \sum _{k_a=n-1}^{T_a-1} \Delta^{(i)[a]} _{k_a} y^{[a]} _{j,k-l}
\\
Y _{j'l'jl} &\equiv \sum _{a=1}^A \sum _{k_a=n-1}^{T_a-1} y^{[a]} _{j, k-l} y^{[a]} _{j', k-l'}
\\
D^{(i')} _{j'l'} &= \sum _{j=1}^N \sum _{l=0}^{n-1} Y _{j'l'jl} \,\chi^{(i')} _{jl}
\\
\chi^{(i')} _{j'l'} &= \sum _{j=1}^N \sum _{l=0}^{n-1} Y^{-1} _{j'l'jl} D^{(i')} _{jl}
\end{aligned}$$

Here, $a$ is the index of data sequence, and $A$ is the number of data sequences.
