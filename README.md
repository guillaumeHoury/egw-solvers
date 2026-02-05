# Gromov-Wasserstein Optimal Transport

Implementation of several Sinkhorn-based Gromov-Wasserstein (GW) solvers.
The file `main.py` provides several code examples.

## Overview

Here are the solvers gradient computations implemented:

### Gromov-Wasserstein Solvers

**Matrix-Based (cubic complexity):**
- `EntropicGW`: Standard entropic GW
- **`KernelGW`: Kernel-based formulation**
- `ProximalGW` / `ProximalKernelGW`: Proximal regularization

**Embedding-Based (quadratic complexity, linear memory footprint):**
- `QuadraticGW`: Solver for squared Euclidean costs
- `LowRankGW`: Low-rank approximation of cost matrices
- `SampledGW`: Stochastic sampling of cost matrix coefficients
- `QuadraticDualGradGW`: Dual gradient descent for squared Euclidean costs
- **`CntGW`: Conditional negative type embeddings**
- `CntDualGradGW`: adaptation of `QuadraticDualGradGW` to our CNT framework

**Multiscale:**
- `MultiscaleQuadraticGW`: Multiscale implementation for squared Euclidean costs
- **`MultiscaleCntGW`: Multiscale implementation for CNT costs**

(Algorithms in bold correspond to our contributions).

### Differentiable Optimal Transport

Compute gradients with respect to point cloud coordinates:
- `gradient_quadraticgw`: Gradients for quadratic GW
- `gradient_kernelgw`: Gradients for kernel-based GW
- `gradient_cntgw`: Gradients through dimensionality reduction

### Applications

Compute gradients with respect to point cloud coordinates:
- `barycenters`: GW barycenters for CNT costs
- `gradient_flows`: GW gradient flows for CNT costs

