# World Models: From Predictive Dynamics to Interactive Simulators

## TL;DR
- A world model is best understood functionally: a learned or specified model predicts consequences of actions, and supports simulated experience, planning, or both; there is no single architecture implied by the term. [1][2]
- The central design split is between compact latent dynamics models built for decision-making and generative models that predict/render future observations; hybrid systems increasingly combine the two. [3][4][5]
- Recent systems broaden the target from offline visual plausibility to controllable, long-horizon interaction in robotics and driving, while evidence shows visual quality does not guarantee task success. [6][7][8]
- Evaluation must test action response, physical and temporal consistency, long-horizon stability, and closed-loop utility—not only frame-level appearance. [9][10][11]
- Reliability under distribution shift, causal interventions, uncertainty, and safety-critical use remain major unresolved issues. [11][12]

## Background
“World model” covers a family of methods rather than one canonical network. In reinforcement learning, an environment model predicts what follows from a state and action, including next state and reward; planning is a computation over a model that produces or improves a policy. The Dyna framework integrates real-world interaction with planning that uses a learned world model. [1][2] Early recurrent-network work also explored learning environmental dynamics and using internal simulation for goal-directed behavior. [13]

The term gained a modern neural-learning interpretation in systems that compress observations, predict latent futures, and provide those representations to a controller. Ha and Schmidhuber’s architecture separates a visual representation, recurrent dynamics, and a controller, illustrating how a learned simulator can provide training experience beyond direct interaction. [3] Across the field, models may be evaluated as predictive representations, simulators, or components of decision systems; conclusions depend on which role is intended. [11]

## 1. Two broad modeling targets: decision-relevant state or rendered future

One approach predicts a compact latent state rather than reconstructing every observation pixel. Dreamer learns an action-conditioned latent dynamics and reward model, then trains actor and value functions on imagined trajectories; this makes imagined rollouts a means of policy improvement rather than merely a forecast display. [4] MuZero takes a different decision-centered route: its learned model is used with tree search and predicts policy, value, and reward quantities useful to planning rather than requiring full observation reconstruction as its planning output. [14]

Reconstruction-free methods such as MuDreamer and UniZero push this distinction further: predictive objectives can emphasize value, actions, reward, or decision-relevant latent transitions without a pixel decoder. [15][16] These methods may avoid spending capacity on visual details irrelevant to control, but their latent states are not necessarily interpretable or sufficient for downstream tasks beyond their training objective. The sources describe distinct tradeoffs, not a universally superior representation.

A complementary direction predicts future video or other observable outputs. This can make a simulator inspectable and provide rich training or evaluation data, but visual realism alone does not establish that the model captures controllable causal dynamics. Recent surveys identify weaknesses in physical-law reproduction, causal reasoning, interventions, and complex long-term simulation despite realistic or coherent clips. [11]

## 2. Learning and using dynamics: imagination, search, and hybrid planning

World models are integrated into control in several ways. Dreamer propagates learning through imagined trajectories to improve actor and value networks. [4] MuZero combines learned dynamics with tree-based search, using the model as a substrate for decision-time exploration. [14] Latent-space planning methods also study direct rollout planning and MCTS for continuous control, where candidate action sampling and deeper search trade computation for look-ahead. [17]

Newer latent systems continue to combine representation learning and planning, while choosing different objectives and control mechanisms. UniZero is a MuZero-style latent planner with transformer-based modeling and MCTS; MuDreamer instead emphasizes predictive representation objectives and Dreamer-like imagined actor-critic learning. [15][16] Their contrast demonstrates that “world model” does not specify whether decisions are learned in imagination, optimized over trajectories, or searched online.

## 3. Embodied and interactive models

For embodied agents, the relevant prediction is often action-conditioned: given a history and an action, what will the robot or scene do? IRASim predicts manipulation video from observation history and robot action trajectories, and reports use in policy evaluation, model-based planning, and controllable virtual-arm interaction. It also reports that generated video can extend beyond 150 frames, while noting generation is not real time. [6] EVA represents another embodied future-video approach, combining video generation and visual-language models. [18]

Interactive video systems attempt to bridge dynamics and rendering. PAN describes an architecture with an autoregressive latent-dynamics backbone conditioned on history and language actions, plus a video-diffusion decoder for detailed observations; its stated goal is long-horizon, controllable simulation. [5] This separation treats latent evolution as the simulation substrate and the video decoder as perceptual rendering. It is a promising architectural pattern, but claims of broad simulation capability require task and intervention tests in addition to generated-video inspection.

## 4. Autonomous driving and policy-coupled prediction

Driving world models face multimodal, spatial, temporal, and control requirements. Recent work includes methods targeting collaborative state-action prediction for trajectory planning, and models that represent driving across RGB, semantics, depth, occupancy, action, and reward. [8][19] These designs reflect a move from generating plausible scenes alone toward connecting scene evolution to candidate behavior and policy evaluation. The sources establish these as proposed approaches, rather than demonstrating a settled benchmark advantage.

Benchmarks are developing alongside models. WorldSimBench evaluates video-generation models from visual and action perspectives, including manipulation-oriented assessment. [9] DrivingGen’s record describes metrics for visual realism, trajectory plausibility, temporal coherence, and controllability, suggesting evaluation is expanding beyond image quality. [20] Such metrics are complementary: scene fidelity, response to controls, and decision utility measure different capabilities.

## 5. Evaluation: from plausible frames to useful closed-loop behavior

A robust evaluation stack should distinguish at least three questions: does the output look plausible; does it obey the action and physical constraints; and does using the model improve task performance in interaction? WorldModelBench targets instruction following and physics adherence, highlighting violations that general video-quality scores may miss. [10] World-in-World evaluates models within closed-loop tasks and reports that visual quality does not guarantee task success. [7]

Survey work distinguishes world models used to represent mechanisms from models evaluated as future-state predictors, so an evaluation should match the model’s intended capability. [11] Neither one metric nor one benchmark covers every intended function. For embodied prediction, action consistency, physical conformity, temporal stability, and closed-loop utility are particularly important complements to appearance-based measures. [7][9][10][11]

## Trends and open problems

The research trajectory is from compact models for control toward richer generative simulators, and increasingly toward systems that combine latent reasoning, action-conditioned prediction, and rendered outputs. [4][15][5][21] Applications in robot manipulation and driving make a key distinction visible: a model may produce convincing observations yet still be unreliable as a decision substrate. Closed-loop evaluation and policy-coupled prediction are therefore central trends, not optional add-ons. [6][7][8]

Several problems remain. First, errors can accumulate over long rollouts, and surveys report difficulty maintaining complex long-term dynamics and physical consistency. [11][12] Second, predictive correlation is not the same as causal competence: models need to respond correctly to interventions and counterfactual actions, rather than merely continue familiar visual patterns. [11] Third, generalization to unseen conditions remains a stated direction for embodied systems, and safety-oriented evaluation remains necessary when model outputs inform downstream actions. [11][7][10]

Finally, model quality must be matched to deployment stakes. A model used for exploratory simulation can tolerate different errors from one used for robotics or autonomous driving; closed-loop benchmarks and physics-adherence assessments provide relevant evidence beyond visual appearance. [7][10] Progress should therefore be measured by reliable prediction, actionable physical grounding, and verified closed-loop benefit—not scale or visual fidelity alone. [11][7][10]

## References
[1] Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. web. http://www.derongliu.org/adp/adp-cdrom/refs/sutton19900216.pdf (1990)
[2] 9.1 Models and Planning. web. http://incompleteideas.net/sutton/book/ebook/node95.html (n.d.)
[3] World Models. web. https://arxiv.org/abs/1803.10122 (2018-05-09)
[4] Dream to Control: Learning Behaviors by Latent Imagination. web. https://arxiv.org/pdf/1912.01603 (n.d.)
[5] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. arxiv. https://arxiv.org/abs/2511.09057 (2025-11-14)
[6] IRASim: A Fine-Grained World Model for Robot Manipulation. web. https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_IRASim_A_Fine-Grained_World_Model_for_Robot_Manipulation_ICCV_2025_paper.pdf (2025)
[7] World-in-World: World Models in a Closed-Loop World. arxiv. https://arxiv.org/abs/2510.18135 (2025-10-20)
[8] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction. arxiv. https://arxiv.org/abs/2510.19654 (2025-10-22)
[9] WorldSimBench: Towards Video Generation Models as World Simulators. hf-search. https://huggingface.co/papers/2410.18072 (2024-10-23)
[10] WorldModelBench: Judging Video Generation Models As World Models. hf-search. https://huggingface.co/papers/2502.20694 (2025-02-28)
[11] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[12] World Models: The Safety Perspective. web. https://arxiv.org/html/2411.07690 (2024)
[13] Making the World Differentiable: On Using Self-Supervised Fully Recurrent Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments. web. https://people.idsia.ch/~juergen/FKI-126-90ocr.pdf (n.d.)
[14] Mastering Atari, Go, chess and shogi by planning with a learned model. web. https://www.nature.com/articles/s41586-020-03051-4 (2020-12-23)
[15] MuDreamer: Learning Predictive World Models without Reconstruction. web. https://arxiv.org/pdf/2405.15083 (2024)
[16] UniZero: Generalized and Efficient Planning with Scalable Latent World Models. web. https://arxiv.org/html/2406.10667 (2024)
[17] Dream and Search to Control: Latent Space Planning for Continuous Control. web. https://arxiv.org/pdf/2010.09832 (2020)
[18] EVA: An Embodied World Model for Future Video Anticipation. hf-search. https://huggingface.co/papers/2410.15461 (2024-10-20)
[19] OmniNWM: Omniscient Driving Navigation World Models. arxiv. https://arxiv.org/abs/2510.18313 (2025-10-21)
[20] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving. hf-search. https://huggingface.co/papers/2601.01528 (2026-01-04)
[21] Simulating the Visual World with Artificial Intelligence: A Roadmap. arxiv. https://arxiv.org/abs/2511.08585 (2025-11-11)
