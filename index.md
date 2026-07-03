---
layout: project_page
permalink: /

title: Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control
conference: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026
authors:
    Keqin Wang$^*$, Tao Zhong$^*$, David Chang, Christine Allen-Blanchette
affiliations:
    Princeton University <br> <small>$^*$Equal contribution</small>
paper: https://arxiv.org/abs/2509.14431
video: https://arxiv.org/abs/2509.14431 # TODO: placeholder — replace with real video
code: https://github.com/CAB-Lab-Princeton/LEGO-MARL
# data: # TODO: add dataset link if released
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Multi-agent reinforcement learning (MARL) has emerged as a powerful paradigm for coordinating swarms of agents in complex decision-making, yet major challenges remain. In competitive settings such as pursuer-evader tasks, simultaneous adaptation can destabilize training; non-kinetic countermeasures often fail under adverse conditions; and policies trained in one configuration rarely generalize to environments with a different number of agents. To address these issues, we propose the Local-Canonicalization Equivariant Graph Neural Networks (LEGO) framework, which integrates seamlessly with popular MARL algorithms such as MAPPO. LEGO employs graph neural networks to capture permutation equivariance and generalization to different agent numbers, canonicalization to enforce $E(n)$-equivariance, and heterogeneous representations to encode role-specific inductive biases. Experiments on cooperative and competitive swarm benchmarks show that LEGO outperforms strong baselines and improves generalization. In real-world experiments, LEGO demonstrates robustness to varying team sizes and agent failure.
        </div>
    </div>
</div>

---

<!-- TODO: add teaser figure / method overview figure here -->

## Overview

Swarm robotics needs control policies that are **sample-efficient** to train, **generalize** across different team sizes, and hold up in the **real world**. Standard MARL policies built on multi-layer perceptrons ignore the combinatorial and geometric structure of these tasks, so they scale poorly as the number of agents grows and rarely transfer to configurations they were not trained on.

**LEGO** (*Local-Canonicalization Equivariant Graph neural networks*) bakes the right inductive biases directly into the policy by decoupling the two symmetries of the problem:

- **Permutation symmetry** is handled by a **graph neural network** over the agents.
- **Euclidean symmetry** — $E(n)$-equivariance — is handled by **local canonicalization** of each agent's inputs.
- **Role structure** (e.g., pursuers vs. evaders) is captured with **heterogeneous graphs**.

Because it is modular, LEGO drops into off-the-shelf MARL algorithms such as **MAPPO**, yielding LEGO-MAPPO.

## Why it's hard

- **Curse of many agents.** MLP policies scale badly as the agent count grows; a policy trained with $N$ agents typically fails when deployed with a different number of agents $M$, forcing costly retraining for every swarm configuration.
- **Sample inefficiency.** MARL often fails to exploit the geometric symmetries inherent in robotics tasks, wasting samples relearning equivalent situations.
- **Competitive instability.** In pursuer–evader settings, the simultaneous adaptation of both sides can destabilize training.

## Method: LEGO

LEGO's design philosophy is to *decouple* the system's symmetries and handle each with the right tool. Each agent first **canonicalizes** its local frame, which removes $E(2)$ nuisance variation before encoding; a role-wise **graph network** then encodes the (sub)graphs into a state representation; actions are produced in the local frame and mapped back to the global frame via the agent's orientation. The whole pipeline is trained end-to-end with **MAPPO** under centralized-training / decentralized-execution.

Because canonicalization removes $E(2)$ variation before encoding, the resulting policy is **$E(2)$-equivariant by construction**, and the graph structure makes it **permutation-equivariant** within each role.

## Contributions

- A novel and **modular framework, LEGO**, that synergistically integrates local canonicalization for geometric $E(n)$-equivariance, graph neural networks for permutation equivariance, and heterogeneous graphs for role-based policies.
- Empirical demonstration that LEGO, when integrated with MAPPO, achieves **superior sample efficiency and performance** over strong baselines in both cooperative and competitive MARL tasks.
- Rigorous validation of **generalization**: zero-shot scalability to unseen numbers of agents, and out-of-distribution generalization to novel geometric configurations.
- **Real hardware demonstration** on Crazyflie drones, highlighting the framework's potential for real-world robotics.

## Results

- Outperforms strong baselines on cooperative and competitive swarm benchmarks (including the MPE Tag pursuer–evader task).
- **Zero-shot** transfer to team sizes never seen during training.
- Robustness to **agent failure** and varying team sizes in **real-world** Crazyflie drone flights.

## Citation

```bibtex
@article{wang2025lego,
  title   = {Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control},
  author  = {Wang, Keqin and Zhong, Tao and Chang, David and Allen-Blanchette, Christine},
  journal = {arXiv preprint arXiv:2509.14431},
  year    = {2025}
}
```
