# spectral operator learning
Here you find the **S**pectral **O**perator **L**earning **U**nder construc**TION**

It's the code for the paper:
- [Render unto Numerics : Orthogonal Polynomial Neural Operator for PDEs with Non-periodic Boundary Conditions](https://arxiv.org/abs/2206.12698)

## Requirements
This program is developed based on pytorch 1.9.0. Older version may not support.

## datasets
The datasets are now available at
- [dataset](https://drive.google.com/drive/folders/1YLsK5GkFpRvrUI4olSEBaz1Jo7T7lO0C?usp=sharing)

except the full dataset for heat-Robin-BC experiment with full (8192+1) resolution and all (10^6) samples, the file size being 24G. If you need the full heat-robin dataset please contact us via email. And thanks for my friend Hui Zhong helping me uploading the datasets.

Each data file is loaded as tensors. The first index is the samples; the rest of indices are the discretization.

For all of the input and output functions, the sampling on Chebyshev-Gauss-Lobatto (CGL) points and uniform points are given. Functions on uniform grids are named by `u0_unif` and `u1_unif`, respectively, while those on CGL points are `u0_cgl` and `u1_cgl`.

- For the heat-Robin-BC(1k) dataset `heat_robin1k` and heat-Dirichlet dataset, the shape of data is in [1100, 8193], with the first 1000 pieces being the training set and the rest test set. 
- For the Burgers-Neumann-BC dataset, the shape is [1100, 4097], due to the limitation of computational resources when generating it. 
- For the heat-Robin-BC(with N=256) `heat_robinN256` dataset, it is of the shape [101000, 256].

## Reproducing the experiments in paper
We fixed the random seeds as 0 in our experiments, so the reproducing should be easy. As for the detailed settings of hyperparameters, users can refer to `~/results/experiment_parameters.py`. The code is for reference only, and we do not recommend to insert them directly, 

## Links
- [shenfun](https://github.com/spectralDNS/shenfun), an excellent PDE solver based on spectral Galerkin methods.
- [chebfun](https://www.chebfun.org/), a famous PDE solver mainly based on spectral collocation method.
- [Fourier Neural Operator](https://github.com/neuraloperator/neuraloperator), the state-of-the-art neural operator for parametric PDEs. 
