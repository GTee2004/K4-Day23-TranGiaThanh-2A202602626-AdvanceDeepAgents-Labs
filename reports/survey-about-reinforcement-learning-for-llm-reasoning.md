# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR
- RL complements supervised fine-tuning (SFT): instead of only imitating demonstrations, it updates a policy from rewards on sampled responses; the resulting behavior depends on what the reward measures and how it is optimized. [1][2]
- The central design axes are feedback source and granularity (verifiable outcome, learned outcome, or process signal) and policy optimization (critic-based PPO versus critic-free group-relative approaches such as GRPO). [2][3]
- Process feedback can improve step-level credit assignment, but results from reward-model selection are not equivalent to proving that PRM-driven policy RL always outperforms outcome-reward RL. [4][5]
- DeepSeek-R1’s report describes a multi-stage reasoning-training recipe; it should not be treated as a pure-RL-only system. [6]
- Gains remain difficult to compare: verifiers, exploration, output length, inference budgets, contamination, and incomplete recipe disclosure all affect results and reproducibility. [7][8][9]

## Background

The familiar RLHF pipeline establishes a useful starting point. It separates supervised demonstrations, preference comparisons used to train a reward model, and policy optimization against the learned scalar reward. In the described setup, PPO samples responses, maximizes reward, and uses a per-token KL penalty relative to the SFT policy to limit reward-model over-optimization. This differs from SFT’s direct imitation objective: policy learning can score sampled completions that were not themselves demonstrations. The reward is not an objective definition of human values; the original authors explicitly delimit it as reflecting their labelers’ and researchers’ preferences. [1]

For reasoning, the prompt can be viewed as context/state and the model’s next-token distribution as its policy. The reward may judge a completed answer, intermediate steps, or broader preferences. A useful survey taxonomy therefore separates (1) where feedback comes from—programmatic verifiers, learned reward models, or people/AI feedback; (2) when it is assigned—outcome-level versus process-level; and (3) how the policy is updated, including critic-based and critic-free methods. These dimensions overlap: a verifier can provide outcome reward, while a method may combine a final correctness signal with step-sensitive feedback. [2][3]

This distinction helps avoid overclaiming what RL contributes. The motivation is not that supervised learning cannot teach reasoning: the surveyed literature instead emphasizes that RL can use trial-and-error reward to improve a specified objective, while high-quality labeled reasoning data and step annotations can be costly. The choice of reward and starting model remains consequential. [2]

## Reward design: verifiable outcomes and process feedback

Outcome reward models (ORMs) score a completed solution, and programmatic verifiable rewards (such as answer checking or code tests) can give relatively objective signals when a task admits a reliable checker. Both concentrate credit at the result, even though their feedback sources differ. Process reward models (PRMs), by contrast, score intermediate reasoning steps and aim to provide denser credit assignment. Learned process rewards may be more nuanced, but can also be noisy or vulnerable to reward hacking; density alone does not guarantee correctness. [2][3]

Empirical evidence supports a qualified case for process supervision. In *Let’s Verify Step by Step*, process-supervised selection outperformed outcome-reward selection in the reported MATH best-of-N setting, and the work released PRM800K, a dataset of step-level human feedback labels. These are reward-model and best-of-N selection results—not a controlled demonstration that every policy trained with a PRM beats RL trained on outcome rewards. [4]

Another study of GSM8K distinguishes answer correctness from trace quality. Outcome supervision achieved similar final-answer error with less label supervision in the reported setup, while process feedback or a reward model emulating it was needed for low reasoning-trace error. In its comparisons, direct RL on final-answer correctness had higher trace error than the best process-based method. Together these results show why evaluation should inspect both final answers and intermediate reasoning, while not assuming that fluent traces are faithful evidence of how a model reasoned. [5][2]

An alternative hybrid appears in Posterior-GRPO for code generation: it gates a learned reasoning reward on task success, combining process and outcome information in an attempt to reduce reward hacking. The Hugging Face paper record reports a 7B model improving 4.5% over outcome-only reward baselines across code benchmarks and generalizing to math tasks. This is a promising design example, not yet a general ranking of process-aware methods. [10]

## Policy optimization: PPO, GRPO, and recipe effects

PPO with a learned value/critic model is a standard RLHF baseline, but token-level value estimation can be difficult under sparse end-of-answer rewards. Surveys classify critic-free group-relative methods such as GRPO alongside critic-based policy optimization. DeepSeekMath presents GRPO as an approach for mathematical reasoning and reports improved results after RL fine-tuning; its system scores do not isolate the algorithm from every other training choice. [2][3]

GRPO’s group comparison is intended to reduce the need to train a separate value model, but sampling and reward quality become central. Surveys distinguish outcome-level reward from explicit process-level rewards and note the associated credit-assignment tradeoff. [2][3]

Later methods modify the recipe rather than simply replacing the reward. DAPO is presented as an open-source RL system with training optimizations for large-scale mathematical reasoning. The HF paper record summarizes its system as using decoupled clipping and dynamic sampling, while the primary report is the relevant source for its complete method details and measured comparisons. This line of work highlights optimization and response-length management as factors, not a universal method ranking across differently configured experiments. [11][12]

Other work explores critic-based PPO, critic-free GRPO variants, and modified reward structures; no single optimizer is established as best across models, tasks, and compute budgets. The broad literature organizes choices across critic use, online versus offline learning, sampling, KL/entropy regularization, and outcome versus process signals. [2][3]

## Recent development: DeepSeek-R1 and open reproduction

DeepSeek-R1-Zero is a prominent demonstration of reasoning-focused RL without an initial SFT stage: the report describes applying GRPO directly to a base model. It also identifies readability and language-mixing weaknesses in R1-Zero. Full DeepSeek-R1 should not be described as the same pure-RL recipe: it adds cold-start long-CoT examples, reasoning RL, rejection-sampled SFT, and another RL phase. Its report also provides decoding details, underscoring that benchmark scores depend on evaluation protocol. [6][13]

The 2025 workstream extends from large-model demonstrations toward recipe disclosure and scaling studies. DAPO presents an open-source system; Open-R1 launched to reconstruct data and training choices and investigate compute/data scaling, while initially describing an intended reproduction plan rather than claiming completed reproduction. These efforts matter because model weights without training code and datasets do not fully specify a reproducible experiment. [11][7][12]

Recent work also tests the boundaries of pure outcome-reward scaling. Exploration research reports that outcome-only RL can improve accuracy while reducing generation diversity, and proposes exploration bonuses to mitigate this effect. The finding makes diversity an important training diagnostic alongside benchmark accuracy: a policy may improve measured correctness while narrowing its sampled solution space. [9]

## Evaluation and evidence quality

Numerical comparisons require matched inference budgets and transparent evaluation settings. A 2025 position paper warns that pass@k results can conflate policy improvement with extra sampling when the RL model receives a larger inference budget. It recommends matched-k comparisons, accuracy-versus-budget curves, run variance, calibration and abstention reporting, contamination screening, and stress tests for LLM judges. It also describes sensitivity to judge prompts, dataset versions, option order, and execution configuration. [8]

Reproduction gaps complicate interpretation. The DeepSeek-R1 paper is a detailed report of a multi-stage system, while Open-R1 explicitly began as an effort to reconstruct data and training pipeline choices not released with the weights. Open-source training systems and HF paper records provide useful access points to implementations and related work, but their summaries should not be mistaken for independent replication evidence. [6][7][12][13]

Claims about reasoning itself need additional caution. Answer accuracy is measurable on checkable tasks, but it does not establish that every generated intermediate step is correct or causally responsible for the answer. Process-supervision studies measure trace quality with their own criteria; surveys also identify model-based reward hacking and unreliable evaluation as persistent concerns. A robust comparison should report answer accuracy, trace quality where appropriate, resource use, sampling budget, calibration, and safety/generalization rather than relying on one headline score. [2][4][5][8]

## Trends and open problems

The field is moving from the coarse question “RL or SFT?” toward joint design of data, verifiers, reward granularity, optimization, and test-time compute. Verifiable outcome reward reduces dependence on human step labels where reliable checking is possible, but leaves open how to train on open-ended tasks where objective verification is unavailable. Process feedback offers finer credit assignment, yet raises annotation and reward-model costs and introduces its own failure modes. A code-generation method summarized in a Hugging Face paper record gates learned reasoning rewards on task success, illustrating one possible hybrid design. [2][3][10]

Exploration and efficiency remain coupled. Diversity loss under outcome-based RL can constrain the range of candidate reasoning paths; long traces and repeated attempts can increase training and inference costs. Methods that reward sufficient early exits or control response length may reduce cost, but length penalties can also suppress useful reasoning. Evaluation must therefore compare quality at matched compute, and report length and diversity rather than treating shorter or longer traces as inherently better. [9][2][8]

The largest unresolved methodological need is reproducible, budget-matched evidence. Training data, seeds, verifier error patterns, reward models, decoding settings, benchmark contamination checks, and compute are all part of the result. Open pipelines can help, but a public weight release alone is not a complete recipe. More broadly, verifiers must be tested for systematic errors, learned rewards for hacking, and reasoning models for calibration, generalization, and safety under distribution shift. The evidence base is promising, but claims of general reasoning gains should remain bounded by task and evaluation protocol. [6][11][7][8][2]

## References
[1] Training language models to follow instructions with human feedback. web. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] A Survey of Reinforcement Learning for Large Reasoning Models. web. https://arxiv.org/html/2509.08827 (n.d.)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. web. https://arxiv.org/html/2402.03300 (2024-04-27)
[4] Let's Verify Step by Step. web. https://arxiv.org/html/2305.20050 (2023-05-31)
[5] Solving math word problems with process- and outcome-based feedback. web. https://ar5iv.labs.arxiv.org/html/2211.14275 (n.d.)
[6] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://ar5iv.labs.arxiv.org/html/2501.12948 (2025-01-22)
[7] Open-R1: a fully open reproduction of DeepSeek-R1. web. https://huggingface.co/blog/open-r1 (2025-01-28)
[8] Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards. arxiv. https://arxiv.org/abs/2509.21882 (2025-09-26)
[9] Outcome-based Exploration for LLM Reasoning. web. https://arxiv.org/abs/2509.06941 (2025-09-08)
[10] Posterior-GRPO: Rewarding Reasoning Processes in Code Generation. hf-search. https://huggingface.co/papers/2508.05170 (2025-08-07)
[11] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. web. https://arxiv.org/html/2503.14476v2 (2025-05-20)
[12] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. hf-search. https://huggingface.co/papers/2503.14476 (2025-03-18)
[13] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-search. https://huggingface.co/papers/2501.12948 (2025-01-22)
