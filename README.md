# 1. Introduction

This project is a linear regression tool to get pseudo-susceptibility. This pseudo-susceptibility $\chi$ is defined as following.
$$\Delta t \cdot \frac{d}{dt}y^{(i)}(t) = \int_{0}^{\infty} dt' \sum_{j=1}^N \;\chi^{(i)}_{(j)}(t')\,y^{(j)}(t - t')$$
Here, $\Delta t$ is a interval time of discrete time sequence.

More strictly, in discrete time, pseudo-susceptibility $\chi$ is defined as
$$ \Delta^{(i)}(t_k) = \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i)}_{(j)}(l\cdot \Delta t) y^{(j)} (t_{k-l}) $$
where,

$$
\begin{aligned}
t_k &= t_0 + k \cdot \Delta t
\\
\Delta^{(i)}(t_k) &= y^{(i)}(t_{k+1}) - y^{(i)}(t_k)
\\
N &= \text{the number of the physical quantities}
\\
y^{(i)}(t_k) &= \text{time sequences of i-th physical quantity}
\\
l &= \text{length to calculate pseudo-susceptibility}
\end{aligned}
$$

# 2. Solving Pseudo-Susceptibility
Define the loss as following
$$ \mathscr{L} = \sum_{i=1}^N \sum_{k=n-1}^{T-1} \left(\Delta^{(i)}_{k} - \sum_{j=1}^N \sum_{l=0}^{n-1}\chi^{(i)}_{jl} y_{j, k-l}\right)^2 $$
where,
$$\begin{aligned}
\Delta^{(i)}_k &= \Delta^{(i)}(t_k)
\\
\chi^{(i)}_{jl} &= \chi^{(i)}_{(j)}(l\cdot\Delta t)
\\
y_{j, k-l} &= y^{(j)}(t_{k-l})
\end{aligned}$$

Find minimum by using derivative:
$$ \frac{\partial \mathscr{L}}{\partial \chi^{(i')}_{j'l'}} = 2\sum_{k=n-1}^{T-1}\left(\Delta^{(i')}_{k} - \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i')}_{jl} y_{j,k-l}\right)\left(-y_{j', k-l'}\right) = 0 $$

Therefore,
$$ \sum_{k=n-1}^{T-1} \Delta^{(i')}_k y_{j',k-l'} = \sum_{k=n-1}^{T-1} \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i')}_{jl} y_{j, k-l} y_{j', k-l'} $$

Here, define new tensors: **D-tensor** and **Y_tensor**
$$\begin{aligned}
D^{(i)}_{jl} &\equiv \sum_{k=n-1}^{T-1} \Delta^{(i)}_k y_{j,k-l}
\\
Y_{j'l'jl} &\equiv \sum_{k=n-1}^{T-1} y_{j, k-l} y_{j', k-l'}
\end{aligned}$$

Then, the final linear equation is the following:
$$ D^{(i')}_{j'l'} = \sum_{j=1}^N \sum_{l=0}^{n-1} Y_{j'l'jl} \,\chi^{(i')}_{jl} $$

Therefore, the pseudo-susceptibility is
$$ \chi^{(i')}_{j'l'} = \sum_{j=1}^N \sum_{l=0}^{n-1} Y^{-1}_{j'l'jl} D^{(i')}_{jl} $$

# 3. w/ Multiple Sequences
If you have multiple time sequences for same system, then you need to change the equations a little bit.

$$\begin{aligned}
\Delta^{(i)}(t^{[a]}_{k_a}) &= \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i)}_{(j)}(l\cdot \Delta t) y^{(j)} (t^{[a]}_{k-l})
\\
\mathscr{L} &= \sum_{a=1}^A \sum_{k_a=n-1}^{T_a-1} \sum_{i=1}^N \left(\Delta^{(i)[a]}_{k} - \sum_{j=1}^N \sum_{l=0}^{n-1}\chi^{(i)}_{jl} y^{[a]}_{j, k-l}\right)^2 
\\
\frac{\partial \mathscr{L}}{\partial \chi^{(i')}_{j'l'}} &= 2 \sum_{a=1}^A \sum_{k_a=n-1}^{T_a-1}\left(\Delta^{(i')[a]}_{k_a} - \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i')}_{jl} y^{[a]}_{j,k-l}\right)\left(-y^{[a]}_{j', k-l'}\right) = 0
\\
\sum_{a=1}^A \sum_{k=n-1}^{T-1} \Delta^{(i')[a]}_{k_a} y^{[a]}_{j',k-l'} &= \sum_{a=1}^A \sum_{k_a=n-1}^{T_a-1} \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i')}_{jl} y^{[a]}_{j, k-l} y^{[a]}_{j', k-l'}
\\
D^{(i)}_{jl} &\equiv \sum_{a=1}^A \sum_{k_a=n-1}^{T_a-1} \Delta^{(i)[a]}_{k_a} y^{[a]}_{j,k-l}
\\
Y_{j'l'jl} &\equiv \sum_{a=1}^A \sum_{k_a=n-1}^{T_a-1} y^{[a]}_{j, k-l} y^{[a]}_{j', k-l'}
\\
D^{(i')}_{j'l'} &= \sum_{j=1}^N \sum_{l=0}^{n-1} Y_{j'l'jl} \,\chi^{(i')}_{jl}
\\
\chi^{(i')}_{j'l'} &= \sum_{j=1}^N \sum_{l=0}^{n-1} Y^{-1}_{j'l'jl} D^{(i')}_{jl}
\end{aligned}$$
Here, $a$ is the index of data sequence, and $A$ is the number of data sequences.