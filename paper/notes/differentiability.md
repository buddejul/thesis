# Differentiability for Functional Delta Method

Different concepts:

- Hadamard
- Gateaux
- Frechet

Some motivations by Shapiro (1991):

- Gateaux derivatives, with respect to the probability measures, often used as heuristic devices (see intuitions in Shapiro (1991)), but not applicable in proofs.
- Frechet is stronger (and sufficient?) but often not applicable in interesting situations
- Hadamard correct in between for statistical applications; some references in the paper.

## Literature

Summary in Shapiro (1990): "On concepts of directional differentiability".

Application to Stochastic Programs in Shapiro (1991): "Asymptotic analysis of stochastic programs".


## Gateaux Differentiability

This is for example introduced in Shapiro (1991).

**Definition (Gateaux Directional Differentiability)**:
For normed spaces $Z, Y$, consider a mapping $g: Z \to Y$.
$g$ is Gateaux directionally differentiable at $\mu \in Z$ if the limit
$$
g_\mu' = \lim_{t \to 0} \frac{g(\mu + t\zeta) - g(\mu)}{t}
$$
exists for all $\zeta \in Z$.

The directional derivative $g_\mu'(\cdot)$ gives a local approximation of the mapping g at $\mu$ in the sense that
$$
g(\mu + \zeta) - g(\mu) = g_\mu'(\zeta) + r(\zeta)
$$
where $r(\zeta)$ is "small" along any fixed direction $\zeta$.
The directional derivative if it exists is positively homogeneous, i.e. $g_\mu'(t\zeta) = tg_\mu'(\zeta)$  (how to show this?).

## Hadamard Differentiability

Notation and definition of Fang and Santos (2019).

In the following, $D$ and $E$ are Banach spaces (*complete normed vector space*), and $\phi: D_\phi\subseteq D \to E$.

**Definition (Hadamard Differentiability)**:
$\phi$ is Hadamard differentiable at $\theta \in D_\phi$ tangentially to a set $D_0\subseteq D$, if there is a continuous linear map $\phi_\theta': D_0 \to E$ such that

$$
lim_{n\to\infty} ||\frac{\phi(\theta + t_n h_n) - \phi(\theta)}{t_n} - \phi_\theta'(h) ||_E = 0,
$$
for all sequences $\{h_n\}\subset D$ and $\{t_n\}\subset R$ sucht that $t_n \to 0, h_n \to h \in D_0$ as $n\to\infty$ and $\theta + t_n h_n \in D_\phi$ for all $n$.

The important point here is the **linearity** of the map $\phi'$, as opposed to directional differentiability, which only requires a continuous map.

**Definition (Hadamard Directional Differentiability)**:
If in addition $g_\mu'(\zeta)$ is linear and continuous in $\zeta$ then $g$ is (fully) Gateaux differentiable at $\mu$.

Shapiro (1991) separately discussed Hadamard (directional) differentiability and (directional) differentiability *tangentially to a set*.

Some relations between Hadamard and Gateaux:

- Hadamard directional differentiability implies Gateaux directional differentiability.
- Hadamard d.d. $\rightarrow$ directional derivaitve $g_\mu'(\zeta)$ is continuous, although possibly nonlinear, in $\zeta$.

Shapiro (1991) introduces the main ideas using Gateaux but says the stronger notion of Hadamard differentiability is required for the delta method.
