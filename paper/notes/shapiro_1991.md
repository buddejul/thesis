# Shapiro 1991: Inference on Optimal Values of Mathematical Programs

The "true" mathematical programming problem (**$P_0$**) is given by:
$$
\min_{x\in S} f_0(x) \\
\text{subject to} \\
f_i(x) = 0, \quad i=1,\ldots, q \\
f_i(x) \leq 0, \quad i=q+1,\ldots, r \\
$$
where $S\subseteq R^k$ and $f_i(x)$ are generally nonlinear, real-valued functions.

We look at the special case where $f_i(x)$ are (affine) linear in $x$.

Shapiro thinks of a sequence of approximating programs that are available to us.
Denote the programsb by **$\hat{P}_n$**.

$$
\min_{x\in S} \psi_{0n}(x) \\
\text{subject to} \\
\psi_{in}(x) = 0, \quad i=1,\ldots, q \\
\psi_{in}(x) \leq 0, \quad i=q+1,\ldots, r \\
$$

Denote the true optimal value and solution by $\rho_0$ and $x_0$, respectively.
The estimators are denoted by $\hat{\rho}_n$ and $\hat{x}_n$.

An example is where the true constraints are expectations over known functions.

## Intuition

Basically a delta method with appropriate derivatives.


## Asymptotic properties of the optimal value

Goal: Study asymptotic behavior of $\hat{\rho}_n$ of the sequence of approximating programs **$\hat{P}_n$**.
