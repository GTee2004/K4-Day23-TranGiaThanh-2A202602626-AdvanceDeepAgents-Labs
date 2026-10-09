# Video and Multimodal Generation: A Survey of Models, Control, and Evaluation

## TL;DR
- Video generation has moved from short, relatively simple clips toward large latent diffusion and transformer-based systems, but “video generation” remains a family of distinct tasks—text-to-video, image-to-video, editing, continuation, and conditional generation—with different control and evaluation needs. [1][2][3][4]
- A central architectural trade-off is between autoregressive generation, which naturally supports sequential construction and extension, and diffusion-style generation, which iteratively denoises video (often in a compressed latent space) but must explicitly manage temporal coherence and computation. [2][3][5]
- Recent open model families expand access to image-to-video and editing, while systems such as Veo and research models such as SyncFlow explore audio-video generation; system and paper-reported results should not be confused with independent comparisons. [6][7][8][9]
- Better control often adds structure beyond text—camera poses, depth, pose, reference frames, or audio—and long-video methods use strategies such as persistent geometry caches or globally normalized controls. [10][11]
- Benchmarks reveal that visual quality and prompt alignment alone do not establish temporal composition, physical plausibility, safety, or trustworthy provenance; evaluation must remain multidimensional. [12][13][14][15][16][17]

## Background
Video generation maps one or more conditions—such as text, an image, a sequence of frames, or audio—to a temporally ordered visual output. The task is harder than producing a plausible still image: generated frames must express the requested scene and actions while preserving identities, objects, and motion coherently over time. Early text-to-video systems commonly produced short and relatively simple scenes; later work broadened both scene complexity and task types. [1]

The field is not one homogeneous benchmark problem. Text-to-video (T2V) starts from language; image-to-video (I2V) animates or extends a still image; video editing changes selected content while preserving the rest; and audio-conditioned or joint audio-video systems connect motion and sound. These tasks differ in what “success” means: a good I2V result may need to preserve the source image, while an editing result must change the requested object without damaging unrelated content. [4][12][13][8]

## From autoregressive frames to latent generative models

Foundational methods illustrate two enduring modeling choices. Video Diffusion Models extended image diffusion to video using a 3D U-Net and temporal attention; spatial and temporal extension procedures were used for longer or higher-resolution generation, including autoregressive use for long videos. [2] In contrast, CogVideo is an autoregressive transformer approach that leveraged a pretrained text-to-image model and hierarchical training for sequential video generation. [3] The broad lesson is not that one paradigm replaces the other: sequential prediction makes temporal construction explicit, while diffusion allows iterative refinement of a whole spatiotemporal representation. [2][3] 

Latent video diffusion addresses the cost of processing every pixel directly. One early approach models video in a low-dimensional 3D latent space and uses hierarchical generation to extend duration; Stable Video Diffusion later describes a training pipeline of text-to-image pretraining, video pretraining, and high-quality video fine-tuning, with temporal layers and image conditioning. [5][4] These designs shift the bottleneck rather than remove it: compression reduces spatial-temporal computation, but duration, resolution, conditioning, and consistency still compete for finite training and inference resources. [5][4]

Diffusion-transformer systems and flow-based formulations extend this toolkit. Wan describes an open suite built around a diffusion transformer and spatiotemporal VAE, with text-to-video, image-to-video, and editing tasks. [6] The examples in current research therefore combine transformer scaling, compressed representations, and task-specific conditioning rather than relying on one fixed architecture. [6][18]

## Control, temporal consistency, and long-form generation

Text prompts are expressive but underspecified for detailed motion or camera behavior. Control methods add explicit inputs such as camera paths, depth, pose, sketches, references, trajectories, or audio; the challenge is to honor those constraints without reducing prompt alignment, visual quality, or temporal consistency. [10] GEN3C makes the geometric route concrete: it describes a depth-derived 3D point-cloud cache rendered using a user-provided camera trajectory to guide video generation. This illustrates how explicit geometric structure can support camera control rather than relying on text alone. [11]

Long-duration generation introduces accumulated drift and uneven control. LongVie is an autoregressive approach that combines dense and sparse controls and reports using unified noise initialization, global normalization of controls, and degradation-aware training to address consistency across clips. Its benchmark contains 100 high-resolution videos of at least one minute, with its evaluation set up around overlapping short clips; the paper reports leading results among its tested baselines, not universal superiority. [10] Such methods make clear that “long video” is not simply a larger short clip: persistent subjects, scene state, and control signals must survive repeated generation steps. [10][11]

Editing requires a separate view of quality. FiVE, for example, is described as an object-level editing benchmark containing real and generated videos, several edit types, prompt pairs, and masks; it measures edit success together with preservation and temporal consistency. This is a more diagnostic framing than assessing a generated result only by its overall visual appeal. [13]

## Audio-video and broader multimodal generation

Multimodal video systems cover several different directions. Joint text-to-audio-video models generate both modalities from a common prompt; video-to-audio methods synthesize sound for an existing clip; audio-to-video methods use sound to guide motion or appearance; and speech-oriented systems may generate voice or dubbing without synthesizing new visual frames. Keeping these task definitions separate prevents over-reading claims about “multimodal” generation as evidence of one unified capability. [8][9][19][20]

SyncFlow is an example of joint generation: it describes separate audio and video diffusion-transformer components followed by joint fine-tuning, and reports temporally synchronized audio-video generation from text. [8] The Veo technical report likewise documents joint diffusion over temporal audio and spatiotemporal video latents, using separate modality autoencoders and a transformer denoiser. [9] AudioGen-Omni describes generation of audio, speech, and song synchronized with input video, while DeepAudio-V1 focuses on video-to-speech/audio with optional text and a multi-stage fusion design. These papers target different outputs and conditions, so their evaluation claims are not directly interchangeable. [19][20]

Product documentation provides evidence of deployed interfaces as well as capability claims. Google's Veo documentation describes native audio generation and an API supporting dialogue prompts, reference images, first/last frames, and video extension; the provider also acknowledges continuing challenges in natural, consistent spoken audio and synchronization. [21] Research and product reports thus show increasing integration of sound and video, but do not yet establish universally reliable dialogue, precise lip synchronization, or independent comparative quality across tasks. [9][21][20]

## Recent systems, openness, and scaling

Recent releases make the technical ecosystem more varied. Wan reports 1.3B- and 14B-parameter diffusion-transformer models and multiple downstream tasks; Open-Sora 2.0 describes a model trained for a reported cost of $200,000 and comparisons based on its own human-preference and VBench evaluations. These demonstrate active work on scale and training efficiency, but the benchmark and cost claims are publisher-reported and bounded by their stated setup. [6][22]

Other systems combine high-capacity architectures with practical efficiency work. HunyuanVideo 1.5 describes an 8.3B-parameter model using selective/sliding-tile attention and a super-resolution network. Its reported architecture and efficiency work illustrates the emphasis on deployment constraints, although those descriptions alone do not establish a cross-system latency or quality ranking. [18] Movie Gen describes a suite for video, synchronized audio, editing, and personalization, reporting a 30B-parameter video model; its record is evidence of the research team's system design, not independent validation of its performance claims. [7]

The landscape now includes open checkpoints and APIs as well as proprietary systems. Open releases enable inspection and local inference but do not by themselves guarantee low operating cost, reproducible comparisons, or parity across all tasks. Proprietary technical reports and model cards can disclose architecture and evaluation details, yet their own provider evaluations should still be distinguished from independent assessment. [6][7][9][18][22]

## Evaluation: perceptual quality is not enough

Video benchmarks increasingly decompose quality into multiple dimensions. VBench++ covers dimensions such as subject consistency, motion smoothness, flicker, and spatial relationships, while extending evaluation toward I2V, long video, and trustworthiness. Its authors use tailored prompts and metrics for individual dimensions, underscoring why a single aggregate score can conceal important differences. [12]

Temporal composition and physical reasoning expose additional failure modes. TC-Bench tests whether a model can render specified transitions between scene states; in its reported experiments, contemporary systems completed fewer than about 20% of its compositional-change cases. VideoPhy tests physical commonsense using human-verified interaction captions and reports that even the best model in its study satisfied both caption adherence and physical judgments on only 19.7% of instances. These are task-specific benchmark results, not general scores for all video generation. [13][17]

VBench-2.0 frames a further gap between superficial faithfulness—plausibility and basic prompt adherence—and intrinsic faithfulness to physics, commonsense, anatomy, and compositional integrity. Alongside automated metrics, the survey and benchmark sources emphasize human preference and specialized evaluation; generic vision-language judges may not reliably assess physical behavior. A credible comparison therefore needs to state its task, sample, metrics, human-review procedure, and known blind spots. [14][17]

## Trends and open problems

Three trends stand out. First, models increasingly combine video with other conditions and outputs: text, images, structured motion, and audio appear in both control pipelines and joint generators. Second, systems use latent representations, diffusion transformers, and more efficient attention or inference designs to manage the high cost of spatiotemporal generation. Third, benchmark design is shifting from general appearance toward long-horizon consistency, fine-grained control, editing preservation, physical plausibility, and trustworthiness. [6][10][12][8][9][18]

Key problems remain open. Maintaining identities, object permanence, scene state, and physically plausible motion over long durations is difficult; adding multiple controls can create conflicts rather than simply improving fidelity. Audio-video systems must be evaluated for timing and semantic alignment as well as audio and visual quality, and speech consistency remains an acknowledged limitation in provider documentation. [10][11][21][14][17]

Safety is also intrinsically temporal: a sequence can be harmful even if individual sampled frames appear benign. T2VSafetyBench evaluates multiple safety categories and reports that no tested system excels across all aspects, while highlighting trade-offs between safety and usability. Provenance records can help document origin and edits; they should be treated as one source of provenance information rather than a complete solution to authenticity or misinformation. Safety controls, provenance, forensic methods, and human judgment are therefore complements—not substitutes. [15][16]

A useful next phase of research should prioritize reproducible cross-system testing, human-aligned multi-axis evaluation, longer coherent outputs, control conflict handling, and evaluation of multimodal synchronization. Reports should separate measured results from provider claims and specify hardware, duration, resolution, and conditioning so that progress in quality can be compared meaningfully with progress in cost and reliability. [12][13][21][14][18]

## References
[1] From Sora What We Can See: A Survey of Text-to-Video Generation. web. https://arxiv.org/abs/2405.10674 (n.d.)
[2] Video Diffusion Models. web. https://arxiv.org/abs/2204.03458 (2022-04-07)
[3] CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers. web. https://arxiv.org/abs/2205.15868 (2022-05-29)
[4] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. web. https://arxiv.org/abs/2311.15127 (2023-11-21)
[5] Latent Video Diffusion Models for High-Fidelity Long Video Generation. hf-search. https://huggingface.co/papers/2211.13221 (2022-11-23)
[6] Wan: Open and Advanced Large-Scale Video Generative Models. arxiv. https://arxiv.org/abs/2503.20314 (2025-03-26)
[7] Movie Gen: A Cast of Media Foundation Models. hf-search. https://huggingface.co/papers/2410.13720 (2024-10-17)
[8] SyncFlow: Toward Temporally Aligned Joint Audio-Video Generation from Text. arxiv. https://arxiv.org/abs/2412.15220 (2024-12-03)
[9] Veo: a text-to-video generation system. web. https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf (2025-05-23)
[10] LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation. web. https://arxiv.org/html/2508.03694v1 (2025-08-05)
[11] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control. web. https://arxiv.org/abs/2503.03751 (2025-03-05)
[12] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. web. https://arxiv.org/html/2411.13503v1 (2024-11-20)
[13] TC-Bench: Benchmarking Temporal Compositionality in Conditional Video Generation. web. https://aclanthology.org/2025.findings-acl.241.pdf (n.d.)
[14] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. web. https://arxiv.org/html/2503.21755 (n.d.)
[15] T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models. web. https://papers.nips.cc/paper_files/paper/2024/file/74eed5f568354c2e77dd9b018f38a9d4-Paper-Datasets_and_Benchmarks_Track.pdf (2024-12-10)
[16] C2PA and Content Credentials Explainer. web. https://spec.c2pa.org/specifications/specifications/2.4/explainer/Explainer.html (n.d.)
[17] VideoPhy: Evaluating Physical Commonsense for Video Generation. web. https://arxiv.org/html/2406.03520v1 (n.d.)
[18] HunyuanVideo 1.5 Technical Report. arxiv. https://arxiv.org/abs/2511.18870 (2025-11-25)
[19] DeepAudio-V1: Towards Multi-Modal Multi-Stage End-to-End Video to Speech and Audio Generation. arxiv. https://arxiv.org/abs/2503.22265 (2025-03-28)
[20] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation. arxiv. https://arxiv.org/abs/2508.00733 (2025-08-01)
[21] Veo 3.1 — Google DeepMind. web. https://deepmind.google/models/veo/ (n.d.)
[22] Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k. hf-search. https://huggingface.co/papers/2503.09642 (2025-03-12)
