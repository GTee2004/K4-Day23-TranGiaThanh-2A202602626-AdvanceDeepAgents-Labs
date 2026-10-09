# Efficient Inference and Small Language Models: A Survey

## TL;DR
- Efficient inference is an end-to-end systems objective: prefill and autoregressive decoding stress different resources, so latency, throughput, memory, and service-level goodput should be measured separately. [1][2][3]
- No compression method is universally best. Quantization reduces bytes and memory, distillation reduces deployed model size, sparsity needs compatible kernels, speculative decoding trades proposal overhead for fewer target-model steps, and KV-cache/serving methods address memory utilization. [4][5][6][7]
- Recent small-model releases emphasize data and post-training quality alongside compact architectures: Phi-3-mini is 3.8B, Gemma 3 spans 1B–27B, and Qwen3 includes dense models from 0.6B plus an MoE model. These are reported capability points, not universal evidence that parameter count alone predicts deployment quality or speed. [8][9][10][11]
- Edge and local deployment can improve locality, privacy, or responsiveness, but the operating point depends on hardware, task, input/output lengths, energy, and fallback costs; quality benchmarks alone cannot determine the best deployment. [12][13][14][15]
- A major open requirement is evaluation on representative devices and workloads, combining task and safety quality with TTFT, decode speed, throughput, memory, energy, and cost; safety should be retested after compression and alignment changes. [12][16][17]

## Background
“Inference efficiency” is not one metric. Transformer inference has a prefill phase, which processes the prompt, and a decode phase, which generates tokens autoregressively. Prefill can be dominated by attention computation for long inputs; decode repeatedly loads weights and grows a key-value (KV) cache as context accumulates. Accordingly, time-to-first-token (TTFT), per-token decode latency, full-request latency, token/request throughput, memory, and power measure different concerns. [1][2]

The system objective matters: offline jobs may prefer throughput, while interactive services prioritize latency targets. Raw peak throughput can be misleading if requests violate service-level objectives; goodput measures the rate of requests that meet those constraints. Model size is only part of the footprint: KV-cache needs and memory fragmentation can limit batch size and context length. [2][3]

Small language models (SLMs) are generally models designed to provide useful capability under tighter compute, memory, energy, and deployment constraints; source conventions differ, and one benchmark study scopes decoder-only SLMs to 100M–5B parameters. “Small” therefore describes an operating regime as much as an absolute parameter cutoff. [18][12]

## Match the optimization to the bottleneck
Surveys identify large model size, attention costs, and sequential decoding as distinct sources of inference inefficiency. A roofline perspective explains why a technique's theoretical operation reduction may not translate to speed: decode is often memory-bound when weights must be fetched at each step, while prefill or large batches can be compute-bound. Quantization is most valuable when reducing memory traffic addresses the active bottleneck, and may help less when compute already dominates. [1][2]

Compression families solve different problems. Post-training quantization changes numeric precision without full retraining; quantization-aware training can adapt weights to quantization effects. Weight-only quantization mainly saves weight storage, while activation and KV-cache quantization target other traffic or memory costs. Low-bit execution may incur dequantization overhead and may not speed up without hardware and kernels that support the representation; quality changes must also be evaluated. [4]

Pruning removes weights or structures, but nominal sparsity is not equivalent to acceleration. Irregular sparse operations can pay indexing costs or map poorly to hardware. SpInfer reports speed gains through a specialized encoding and GPU kernel, illustrating the role of software/hardware co-design rather than a guaranteed gain from pruning itself. Distillation instead transfers teacher behavior to a smaller student, moving substantial cost into training and data preparation to reduce the recurring inference footprint. [4][7][6]

## Serving and decoding: utilization, speculation, and context
PagedAttention addresses KV-cache allocation rather than model weights: it stores cache blocks non-contiguously and allocates them on demand, reducing fragmentation and enabling sharing. Its vLLM implementation reports 2–4× serving throughput over FasterTransformer and Orca at the same latency in the tested settings. This is a system result that combines paging, cache management, and scheduling, not an invariant speedup from paging alone. [19]

Speculative decoding proposes multiple tokens with a cheaper draft process and verifies them in parallel with the target. When accepted, this reduces sequential target-model decode steps while preserving the target distribution under the verification procedure. Online behavior depends on acceptance rate, request load, proposal length, and draft overhead: SmartSpec reports up to 3.2× lower average request latency in its evaluated settings, but fixed speculation can hurt when loads rise or acceptance is poor. [5]

Long-context optimization increasingly focuses on KV-cache size and retention as well as weights. Methods include quantization, pruning/eviction, and latent representations. For example, a small-model study of 30M-parameter GPT models trained on synthetic stories reports 45% KV-cache reduction for half-rank latent multi-head attention with a 0.3% validation-loss increase, while explicitly leaving broader pretrained-model generalization unresolved. Such evidence is promising but should not be generalized beyond its tested models and data. [20]

## SLM capability: recipes matter beyond parameter count
Recent SLMs combine compact architectures with substantial data and post-training. Phi-3-mini is reported as a 3.8B model trained on 3.3T tokens with filtered web and synthetic data; its technical report gives 69% MMLU. Phi-4 is described as a 14B model with an emphasis on data quality, synthetic data, curriculum, and post-training. These reports suggest capability per parameter can be influenced by training recipe, while reported benchmark scores remain tied to the authors' evaluations. [8]

The model landscape is not a single dense-text design. Gemma 3 spans 1B–27B, incorporates vision, multilingual support, and contexts of at least 128K tokens; its report describes local and global attention layers as a way to reduce long-context KV-cache use, and reports distillation across the family. Phi-4-Mini (3.8B) highlights high-quality and synthetic data plus group-query attention, while its multimodal counterpart uses modality-specific LoRA adapters. Qwen3 combines dense sizes from 0.6B with a 30B-A3B MoE model and offers thinking/non-thinking modes with a controllable inference-time thinking budget. [9][21][10]

These approaches expose tradeoffs: long context and multimodality broaden use cases but add memory and systems demands; adaptive reasoning can allocate compute selectively, but its value depends on task and latency budget. An arXiv study reports that same-size model architectures may differ in latency by up to 3.5× and introduces inference-aware scaling; this reinforces that parameter count alone is a poor proxy for runtime. [10][11]

## Edge and hybrid deployment; measurement practice
SLMs are being evaluated on smartphones, edge boards, and distributed edge systems, not only accelerators in datacenters. Benchmarks of dozens of publicly accessible SLMs report that runtime depends on architecture as well as size, and identify prompt processing, token generation, memory, and device variation as critical measurements. Local execution may reduce data movement, latency, or cloud expense; cloud fallback or SLM–LLM collaboration can handle tasks beyond local capacity. [13][14][15]

Fair deployment comparisons need named hardware and runtime, representative prompts and outputs, phase-specific latency, memory, energy, and quality. One SLM benchmark standardized prompt and generation lengths and controlled runtime to isolate model differences; another line of work emphasizes energy/query and cost/query alongside response quality. Throughput gains should be considered with latency distributions or SLO-compliant goodput, particularly for online serving. [12][13][3]

Hugging Face paper records provide useful discovery-level summaries of this landscape, including SLM surveys, Phi-3, Gemma 3, and Qwen3. They complement rather than replace the underlying reports: summaries identify model scope and headline changes, while technical reports or measured studies are needed for detailed claims and their experimental limitations. [22][23][24][25][18]

## Trends and open problems
The field is shifting from isolated compression ratios toward workload-conditioned system outcomes. A smaller weight file does not establish lower end-to-end latency or energy: kernels, memory hierarchy, cache behavior, batch shape, context length, hardware, and serving load all intervene. Cross-paper comparisons remain confounded by differing models, datasets, budgets, and serving stacks, motivating reproducible evaluations that report both quality and realized systems effects. [1][4][12]

Local deployment raises a coupled capacity, privacy, and cost question rather than an automatic cloud replacement. Dynamic routing or escalation can reserve stronger models for difficult requests, but heterogeneous devices, unstable networks, personalization, and semantic consistency complicate hybrid systems. The literature calls for standardized edge–cloud benchmarks that reflect user/device partitioning and non-IID workloads. [13][15]

Safety and capability should be re-evaluated after efficiency changes. An empirical study reports jailbreak susceptibility across evaluated SLMs and examines risks associated with compression, quantization, and distillation. Conversely, EASE proposes selectively activating safety reasoning for adversarial queries; it reports reduced attack success and inference overhead in its experiments. These findings show both the safety risk of constrained models and a possible selective-compute response, not a universal safety guarantee. [16][17]

Open priorities include broader hardware and workload coverage, robust energy and cost accounting, reproducibility of tail latency and goodput, reliable local/cloud routing, and post-transformation safety and privacy testing. More work is also needed to establish when architecture-aware scaling, KV-cache methods, distillation, and adaptive reasoning generalize beyond their reported configurations. [11][12][15][17][20]

## References
[1] A Survey on Efficient Inference for Large Language Models. web. https://arxiv.org/pdf/2404.14294 (2024-07-19)
[2] Efficiently Scaling Transformer Inference. web. https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf (n.d.)
[3] Taming the Titans: A Survey of Efficient LLM Inference Serving. web. https://aclanthology.org/2025.inlg-main.32.pdf (n.d.)
[4] A Survey on Model Compression for Large Language Models. web. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482/A-Survey-on-Model-Compression-for-Large-Language (2024-11-27)
[5] Optimizing Speculative Decoding for Serving Large Language Models Using Goodput. web. https://arxiv.org/pdf/2406.14066v2.pdf (n.d.)
[6] MiniLLM: On-Policy Distillation of Large Language Models. web. https://arxiv.org/abs/2306.08543 (n.d.)
[7] SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs. web. https://cse.hkust.edu.hk/~weiwa/papers/eurosys25-fall-spinfer.pdf (2025-03-30)
[8] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. web. https://www.microsoft.com/en-us/research/publication/phi-3-technical-report-a-highly-capable-language-model-locally-on-your-phone/ (2024-08-30)
[9] Gemma 3 Technical Report. web. https://arxiv.org/html/2503.19786 (2025-03-25)
[10] Qwen3 Technical Report. web. https://arxiv.org/html/2505.09388v1 (2025-05-14)
[11] Scaling Inference-Efficient Language Models. arxiv. https://arxiv.org/abs/2501.18107 (2025-01-30)
[12] Small Language Models: Survey, Measurements, and Insights. web. https://arxiv.org/abs/2409.15790 (2025-02-26)
[13] Edge-First Language Model Inference: Models, Metrics, and Tradeoffs. web. https://arxiv.org/html/2505.16508v2 (n.d.)
[14] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/2025.acl-long.718/ (n.d.)
[15] A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness. web. https://arxiv.org/html/2510.13890v2 (n.d.)
[16] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models. web. https://arxiv.org/abs/2502.19883 (n.d.)
[17] EASE: Practical and Efficient Safety Alignment for Small Language Models. web. https://arxiv.org/abs/2511.06512 (2025-11-09)
[18] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[19] Efficient Memory Management for Large Language Model Serving with PagedAttention. web. https://arxiv.org/abs/2309.06180 (2023-09-12)
[20] Latent Multi-Head Attention for Small Language Models. arxiv. https://arxiv.org/abs/2506.09342 (2025-06-11)
[21] Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs. web. https://arxiv.org/html/2503.01743 (2025-03-03)
[22] A Survey on Efficient Inference for Large Language Models. hf-search. https://huggingface.co/papers/2404.14294 (2024-04-22)
[23] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. hf-search. https://huggingface.co/papers/2404.14219 (2024-04-22)
[24] Gemma 3 Technical Report. hf-search. https://huggingface.co/papers/2503.19786 (2025-03-25)
[25] Qwen3 Technical Report. hf-search. https://huggingface.co/papers/2505.09388 (2025-05-14)
