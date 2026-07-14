---
layout: project_page
permalink: /

title: "LEGO: Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control"
conference: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026
authors:
    Keqin Wang$^*$, <a href="https://taozhong.info/">Tao Zhong</a>$^*$, David Chang, <a href="https://cablanc.github.io/">Christine Allen-Blanchette</a>
affiliations:
    Princeton University <br> <small>$^*$Equal contribution</small>
paper: https://arxiv.org/abs/2509.14431
video: https://arxiv.org/abs/2509.14431 # TODO: placeholder — replace with real video
code: https://github.com/CAB-Lab-Princeton/LEGO-MARL
# data: # TODO: add dataset link if released
---

<!-- Intro figure -->
<div class="columns is-centered">
    <div class="column is-full has-text-centered">
        <img src="/static/image/intro.png"
             alt="An example of equivariance in the MPE Tag environment: pursuers chase evaders while passing through obstacles; permuting agent indices permutes the optimal actions, and rotating agent positions rotates the optimal actions."
             style="width:100%; max-width:900px;">
        <p class="is-size-6 has-text-justified" style="max-width:900px; margin:0.9rem auto 0 auto;">
            <b>Figure 1.</b> An example of equivariance in the MPE Tag environment where <span style="background-color:#D59796; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">pursuers</span> chase <span style="background-color:#9A97BE; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">evaders</span> while passing through <span style="background-color:#CCCCCC; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">obstacles</span>. <em>(Middle to left)</em> As the agents (circles) are permuted by swapping their indices ($s\in S_2$), the optimal actions (arrows) are permuted in the same way. <em>(Middle to right)</em> As the agent positions are rotated $90^\circ$ ($g\in SO(2)$), the optimal actions are also rotated.
        </p>
    </div>
</div>

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

<!-- Pipeline / architecture figure -->
<div class="columns is-centered">
    <div class="column is-full has-text-centered">
        <img src="/static/image/pipeline.png"
             alt="Overview of the LEGO architecture: observations are canonicalized, decomposed into role-specific heterogeneous graphs (self, pursuers, evaders, obstacles), encoded by an MLP or stacked Graphormer layers with pooling, then combined and passed to the actor and critic MLPs.">
        <p class="is-size-6 has-text-justified" style="max-width:900px; margin:0.9rem auto 0 auto;">
            <b>Figure 2. Using LEGO in MARL.</b> For each agent $i$, its raw observation $O_i$ is canonicalized into the local frame $\mathcal{C}(O_i)=\{v_i',\rho(g_i^{-1})O_i\}$. Role-based subgraphs (e.g., <span style="background-color:#D7E5C9; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">self</span>, <span style="background-color:#D59796; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">pursuers</span>, <span style="background-color:#9A97BE; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">evaders</span>, <span style="background-color:#CCCCCC; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">obstacles</span>) are encoded with Graphormer, pooled by role, and concatenated into $s_i$. The policy outputs a local action $a_i^{\text{loc}} \sim \pi_\theta(s_i)$, which is transformed back via $a_i = R_i a_i^{\text{loc}}$. Under CTDE, the critic uses the same pipeline but operates on the global state $X$ during training to obtain $s_i^{\text{global}}$, from which individual values are estimated as $V_i = V_\phi(s_i^{\text{global}})$.
        </p>
    </div>
</div>

## Results

### Cooperative Tasks: MPE Spread<sup>[1]</sup>

Qualitative rollouts on the cooperative MPE Spread benchmark, where agents must spread out to cover all landmarks while avoiding collisions. LEGO-MAPPO is compared against MAPPO, PIC, and SEGNN at two team sizes.

<div class="results-split">
    <div class="results-group">
        <div class="results-group-title">3 Agents, 3 Landmarks</div>
        <div class="results-grid">
            <div class="result-cell">
                <img src="/figures/render_LEG-MAPPO_3agents.gif" alt="LEGO-MAPPO on MPE Spread with 3 agents and 3 landmarks">
                <div class="result-label">LEGO-MAPPO</div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_MAPPO_3agent.gif" alt="MAPPO on MPE Spread with 3 agents and 3 landmarks">
                <div class="result-label">MAPPO<sup>[2]</sup></div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_pic_3agent.gif" alt="PIC on MPE Spread with 3 agents and 3 landmarks">
                <div class="result-label">PIC<sup>[3]</sup></div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_SEGNN_3agent.gif" alt="SEGNN on MPE Spread with 3 agents and 3 landmarks">
                <div class="result-label">SEGNN<sup>[4]</sup></div>
            </div>
        </div>
    </div>
    <div class="results-group">
        <div class="results-group-title">6 Agents, 6 Landmarks</div>
        <div class="results-grid">
            <div class="result-cell">
                <img src="/figures/render_LEG-MAPPO_6agents.gif" alt="LEGO-MAPPO on MPE Spread with 6 agents and 6 landmarks">
                <div class="result-label">LEGO-MAPPO</div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_MAPPO_6agent.gif" alt="MAPPO on MPE Spread with 6 agents and 6 landmarks">
                <div class="result-label">MAPPO<sup>[2]</sup></div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_pic_6agents.gif" alt="PIC on MPE Spread with 6 agents and 6 landmarks">
                <div class="result-label">PIC<sup>[3]</sup></div>
            </div>
            <div class="result-cell">
                <img src="/figures/render_SEGNN_6agent.gif" alt="SEGNN on MPE Spread with 6 agents and 6 landmarks">
                <div class="result-label">SEGNN<sup>[4]</sup></div>
            </div>
        </div>
    </div>
</div>

<div class="results-refs">
    [1] Terry et al. "PettingZoo: Gym for multi-agent reinforcement learning." <i>NeurIPS</i> 2021.<br>
    [2] Yu et al. "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games." <i>NeurIPS</i> 2022.<br>
    [3] Liu et al. "PIC: Permutation Invariant Critic for Multi-Agent Deep Reinforcement Learning." <i>CoRL</i> 2020.<br>
    [4] Chen et al. "E(3)-Equivariant Actor-Critic Methods for Cooperative Multi-Agent Reinforcement Learning." <i>ICML</i> 2024.
</div>

### Competitive Tasks: MPE Tag-occlusion

<div class="results-subtitle">
    2 <span style="background-color:#9A97BE; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">evaders</span> chased by 3 slower <span style="background-color:#D59796; color:#000; padding:0.1em 0.35em; border-radius:0.25em;">pursuers</span>
</div>

<div class="results-row cols-4 dividers">
    <div class="result-cell">
        <img src="/figures/render_LEG-MAPPO_tag_ep2.gif" alt="LEGO-MAPPO on MPE Tag-occlusion">
        <div class="result-label">LEGO-MAPPO</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_MAPPO_tag_ep2.gif" alt="MAPPO on MPE Tag-occlusion">
        <div class="result-label">MAPPO</div>
    </div>
    <div class="result-cell">
        <div class="result-placeholder">Coming soon</div>
        <div class="result-label">PIC</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_SEGNN_tag_small.gif" alt="SEGNN on MPE Tag-occlusion">
        <div class="result-label">SEGNN</div>
    </div>
</div>

### Zero-shot Scalability

Because LEGO is built on a graph neural network, it naturally handles a varying number of input nodes, so a policy trained at one team size can be applied directly to another. We train LEGO-MAPPO on MPE Spread with **4 agents and 4 landmarks**, then evaluate that same policy — with no retraining or fine-tuning — on systems with **2, 3, 5, and 6 agents**.

<div class="results-row cols-4">
    <div class="result-cell">
        <img src="/figures/render_generalization_4to2.gif" alt="LEGO-MAPPO trained on 4 agents, evaluated zero-shot on 2 agents">
        <div class="result-label">2 agents</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_generalization_4to3.gif" alt="LEGO-MAPPO trained on 4 agents, evaluated zero-shot on 3 agents">
        <div class="result-label">3 agents</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_generalization_4to5.gif" alt="LEGO-MAPPO trained on 4 agents, evaluated zero-shot on 5 agents">
        <div class="result-label">5 agents</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_generalization_4to6.gif" alt="LEGO-MAPPO trained on 4 agents, evaluated zero-shot on 6 agents">
        <div class="result-label">6 agents</div>
    </div>
</div>

### Curriculum Learning

LEGO's scalability also enables a curriculum: a policy trained on a smaller swarm serves as a warm start for a larger, harder configuration. Here LEGO-MAPPO is **pre-trained on the 4-agent system** and then trained on the target systems of **6, 7, and 8 agents**.

<div class="results-row cols-3">
    <div class="result-cell">
        <img src="/figures/render_generalization_4to6_curr.gif" alt="LEGO-MAPPO pre-trained on 4 agents, then trained on 6 agents">
        <div class="result-label">6 agents</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_generalization_4to7_curr.gif" alt="LEGO-MAPPO pre-trained on 4 agents, then trained on 7 agents">
        <div class="result-label">7 agents</div>
    </div>
    <div class="result-cell">
        <img src="/figures/render_generalization_4to8_curr.gif" alt="LEGO-MAPPO pre-trained on 4 agents, then trained on 8 agents">
        <div class="result-label">8 agents</div>
    </div>
</div>

### Real-World Experiments

LEGO-MAPPO is deployed on Crazyflie drones, remaining robust to varying team sizes and to agents failing mid-flight.

<!-- TODO: add the failure-experiment video here -->


## Citation

```bibtex
@article{wang2025lego,
  title   = {Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control},
  author  = {Wang, Keqin and Zhong, Tao and Chang, David and Allen-Blanchette, Christine},
  journal = {arXiv preprint arXiv:2509.14431},
  year    = {2025}
}
```
