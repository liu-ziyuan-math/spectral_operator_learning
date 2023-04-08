# Reproduction
Ready-to-reproduce code for the numerical experiments in the paper

- [Render unto Numerics : Orthogonal Polynomial Neural Operator for PDEs with Non-periodic Boundary Conditions](https://arxiv.org/abs/2206.12698)

## Tips
- The [pre-trained models](https://drive.google.com/drive/folders/1YLsK5GkFpRvrUI4olSEBaz1Jo7T7lO0C?usp=sharing) are available.
  - When running the program, if user sets the parameter `epochs` as 0, the program will automatically load the pre-trained models as long as they have been saved under the path `~/model/`
  - Luckily, we conducted the experiments with the random seeds fixed as 0, so the results should be *perfectly identical*， though different seeds should yeild results with negligible difference.
- Choose the `sub` in the `sub_list` to alter the sub-experiment you may want to reproduce.

## Correspondence
- `exp1_burgers_neumann.py` Experiment 1: Viscous Burgers equation with Neumann BCs.
- `exp2_heat_robin.py` the comparison with other deep-learning methods based on the **1k-sized** dataset in Experiment 2: Heat diffusion equation with Robin BCs.
- `exp2-100k_heat_robin.py` the comparison with numerical method based on the **100k-sized** dataset in Experiment 2: Heat diffusion equation with Robin BCs.
- `exp3_heat_inho.py` Experiment 3: Heat diffusion equation with inhomogeneous Dirichlet BCs.
- `exp4_burgers2d.py` Experiment 4: 2D Burgers’ equation with Neumann BCs.
