# LLM Agents and Tool Use: A Survey

## TL;DR
- Tool use extends an LLM with external capabilities—such as search, calculators, browsers, databases, and APIs—so the system can retrieve current information, execute operations, and delegate specialized work beyond text generation alone. [1][2][3]
- Two influential design patterns are modular routing to specialist tools and iterative reasoning–action loops, in which observations from tool execution inform the next step. [2][4]
- Methods range from learning when and how to call APIs, to retrieval-grounded API selection and explicit search over multi-call workflows; each addresses different trade-offs in scale, adaptability, and planning. [3][5][6]
- Evaluation has moved beyond whether a call is syntactically valid: interactive benchmarks measure task completion and trace-level behavior, while newer work highlights safety, evaluator quality, and real deployment constraints. [7][8][9][10][11]
- A persistent gap is dependable operation under uncertainty: agents must select tools appropriately, interpret results faithfully, recover from failures, and do so within acceptable cost and latency. [8][12][13][14]

## Background
An LLM agent is a language-model-driven system that takes observations, chooses actions, and may interact repeatedly with an environment. Tool use is the agent’s ability to invoke external functions or services as part of this process. This distinction matters: ordinary generation produces a response from model context, whereas a tool-using workflow can query changing sources, perform symbolic operations, and observe effects. MRKL frames such systems as a router connected to specialist modules, while WebGPT illustrates a browser-mediated interaction loop that accumulates quotations for a referenced answer. [1][2]

The core control loop is often described as deciding whether to act, selecting a tool and its arguments, receiving an observation, and then continuing or stopping. ReAct made the coupling between reasoning and actions explicit: internal reasoning can maintain or revise a plan, while external actions gather information or alter an environment. This is complementary to the broader agent architecture: planning and memory help sustain behavior across steps, as illustrated by Generative Agents’ use of a memory stream, retrieval, reflection, and decomposed plans. [4][15]

## From modular tools to learned API use
Early systems treated tools as explicit modules with defined interfaces. MRKL proposes a modular system that routes between a language model and external knowledge or discrete reasoning modules, making specialization part of the architecture rather than expecting the language model to do everything internally. WebGPT instead focuses on browser interaction: the model repeatedly acts on a summarized browser state and retains quotations as evidence for final answers. These approaches show two distinct roles for tools—computation and structured knowledge access on one hand, and interactive information gathering on the other. [1][2]

Toolformer shifts the emphasis toward learning tool-use decisions, including which APIs to call and how to incorporate their results into future-token prediction. Gorilla combines API-focused fine-tuning with retrieved documentation, aiming to handle a large and changing API space. Compared with fixed routing or a single generated call, these approaches seek broader API coverage, but depend on correct documentation retrieval and call interpretation. [3][5]

## Interactive control and multi-step orchestration
A one-shot call is not enough when the next action depends on an unpredictable result. ReAct interleaves reasoning with actions and returned observations, allowing an agent to update its plan based on what actually happened. In the cited interactive experiments, ReAct outperformed imitation- or reinforcement-learning methods by 34 percentage points on ALFWorld and 10 points on WebShop. These results illustrate both the value and the context dependence of closed-loop control. [4]

For broader workflows, ToolLLM couples API retrieval with ToolBench solution paths and depth-first search over alternative reasoning paths. The associated dataset covers 16,464 REST APIs across 49 categories; its search procedure can continue promising paths or abandon them for alternatives. This provides an explicit contrast with greedy trajectory generation, but it also adds planning and search overhead. At the system level, TaskMatrix.AI presents a vision of a foundation model that outlines a solution, an API selector that identifies services, and an executor that runs generated actions. [6]

## What benchmarks measure—and miss
AgentBench evaluates agents across eight interactive environments, including operating systems, databases, knowledge graphs, shopping, and browsing. It measures task completion across interactive environments. Its findings emphasize that agent performance involves long-horizon instruction following and environment interaction, not just generating a plausible tool call. [7][16]

The evaluation target is also shifting from final answers to complete trajectories. A survey of agent evaluation argues for measures of intermediate decisions, tool selection, token use, API expense, inference time, policy compliance, and robustness. AgentRewardBench examines automatic evaluation of web-agent trajectories and reports that no single tested LLM judge is best on every benchmark; it also finds rule-based evaluation can underreport success. Taken together, these sources caution that final-answer correctness and automated judging each provide incomplete views of agent behavior. [8][9]

## Safety, robustness, and deployment evidence
Tool-enabled agents inherit risks from both model outputs and the external services they can access. Agent-SafetyBench evaluates safety in interactive settings across 349 environments and 2,000 test cases in eight safety categories. MCP-SafetyBench focuses on real-world Model Context Protocol servers and multi-turn workflows, addressing risks associated with open server connections and interactions across servers. These newer benchmarks broaden safety assessment from isolated prompts toward actions and environment context. [10][11]

Deployment evidence points to a practical preference for constrained workflows. A study spanning 20 case studies and a survey of 306 practitioners across 26 domains reports that 68% of deployed agents execute at most 10 steps before human intervention, and 74% rely primarily on human evaluation. These are findings from that study rather than universal rates, but they underline the gap between ambitious open-ended autonomy and controlled production practice. [12]

## Efficiency and workflow engineering
Each additional model call, tool interaction, and verification step can consume time and resources, making orchestration a systems problem as well as a reasoning problem. Murakkab proposes declarative workflow descriptions, profile-guided optimization, and adaptive runtime configuration; the paper reports reductions in GPU use, energy, and cost against its baselines while maintaining stated service-level objectives. Such results are specific to its evaluated setup, but they exemplify a growing effort to optimize entire workflows rather than a model call in isolation. [17]

Caching and reuse are another route to efficiency. Hugging Face’s record for “Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching” describes adapting structured plan templates from similar tasks, while its record for “What Limits Agentic Systems Efficiency?” describes caching with speculative execution to reduce latency in web-interactive systems without degrading performance. These records point to promising directions, but the available summaries do not supply comparative figures, so they should be read as approach descriptions rather than quantified evidence. [13][14]

## Trends and open problems
The field’s progression is from tool access as a modular add-on, through learned call selection and interleaved action loops, toward larger API ecosystems and end-to-end workflow orchestration. The emerging emphasis is not simply “more tools”: it is grounding actions in documentation, managing multi-step dependencies, observing outcomes, and evaluating safety and efficiency over the whole trajectory. [2][3][5][6][18]

Several problems remain central. First, robustness requires detecting invalid calls, stale or misleading tool outputs, and unexpected environment behavior, then recovering rather than compounding the error. Second, evaluation needs reproducible, trajectory-level diagnostics that distinguish a poor tool choice from faulty execution, inadequate verification, or a flawed judge. Third, safety policies must hold across tools and servers, particularly when actions have side effects. Finally, systems need efficient orchestration and better production observability without relying on a human to intervene after a short sequence. Recent evaluation surveys and deployment studies identify gaps in robustness, scalable evaluation, cost-efficiency, safety, recovery, and observability. [8][9][12][10][11]

The practical design implication is to match autonomy to verifiability: use structured action schemas, retrieved and current tool documentation, explicit observation and verification steps, bounded workflows, and human oversight where consequences or uncertainty warrant it. These are not a settled universal recipe; they are consistent responses to the limits exposed across tool-learning methods, interactive benchmarks, safety evaluations, and deployment reports. [4][5][7][8][12][11]

## References
[1] WebGPT: Browser-assisted question-answering with human feedback. web. https://arxiv.org/abs/2112.09332 (n.d.)
[2] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. web. https://arxiv.org/abs/2205.00445 (n.d.)
[3] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://arxiv.org/abs/2302.04761 (2023-02-09)
[4] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[5] Gorilla: Large Language Model Connected with Massive APIs. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/e4c61f578ff07830f5c37378dd3ecb0d-Paper-Conference.pdf (n.d.)
[6] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. web. https://arxiv.org/abs/2307.16789 (n.d.)
[7] AgentBench: Evaluating LLMs as Agents. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/e9df36b21ff4ee211a8b71ee8b7e9f57-Paper-Conference.pdf (n.d.)
[8] A Survey on Evaluation of LLM-based Agents. web. https://arxiv.org/html/2503.16416v2 (n.d.)
[9] AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. web. https://arxiv.org/abs/2504.08942 (2025-04-11)
[10] Agent-SafetyBench: Evaluating the Safety of LLM Agents. arxiv. https://arxiv.org/abs/2412.14470 (2024-12-19)
[11] MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers. arxiv. https://arxiv.org/abs/2512.15163 (2025-12-17)
[12] Measuring Agents in Production. web. https://arxiv.org/html/2512.04123v2 (n.d.)
[13] Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching. hf-search. https://huggingface.co/papers/2506.14852 (2025-06-17)
[14] What Limits Agentic Systems Efficiency?. hf-search. https://huggingface.co/papers/2510.16276 (2025-10-18)
[15] Generative Agents: Interactive Simulacra of Human Behavior. web. https://arxiv.org/abs/2304.03442 (2023-08-06)
[16] AgentBench: Evaluating LLMs as Agents. hf-search. https://huggingface.co/papers/2308.03688 (2023-08-07)
[17] Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms. web. https://arxiv.org/html/2508.18298 (n.d.)
[18] Large Language Model Agent: A Survey on Methodology, Applications and Challenges. web. https://arxiv.org/html/2503.21460 (n.d.)
