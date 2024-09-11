# 1. Introduction

This project is a linear regression tool to get pseudo-susceptibility. This pseudo-susceptibility $\chi$ is defined as following.
$$\Delta t \cdot \frac{d}{dt}y^{(i)}(t) = \int_{-\infty}^{t} dt' \;\chi^{(i)}_{(j)}(t')\,y^{(j)}(t - t')$$
Here, $\Delta t$ is a interval time of discrete time sequence.

More strictly, in discrete time, pseudo-susceptibility $\chi$ is defined as
$$ \Delta^{(i)}(t_k) = \sum_{j=1}^N \sum_{l=0}^{n-1} \chi^{(i)}_{(j)}(l\cdot \Delta t) y^{(j)} (t_{k-l}) $$
where,
$$\begin{aligned}
t_k &= t_0 + k \cdot \Delta t
\\
\Delta^{(i)}(t_k) &= y^{(i)}(t_{k+1}) - y^{(i)}(t_k)
\\
N &= \text{the number of the physical quantities}
\\
y^{(i)}(t_k) &= \text{time sequences of i-th physical quantity}
\\
l &= \text{length to calculate pseudo-susceptibility}
\end{aligned}$$
