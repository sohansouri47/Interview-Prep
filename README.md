# AMD Inference Engineer Interview Prep Checklist

**Total Topics: 120+**  
**Priority Breakdown: 45 MUST KNOW | 40 SHOULD KNOW | 35 SKIP**

---

## 🔴 MUST KNOW (45 topics) — These Will Definitely Come Up

### vLLM Core Engine (10 topics)
- [ ] **PagedAttention** — Why memory fragmentation was a problem. How KV cache is stored in non-contiguous pages. Page table concept.
- [ ] **Continuous batching** — Why static batching wasted GPU time. How new requests join mid-flight. Iteration-level scheduling.
- [ ] **Continuous batching vs external scheduler** — vLLM batches reactively. Your scheduler shapes input proactively. Different layers.
- [ ] **Prefix caching** — When it helps (shared system prompts). When it does NOT help (random unique prompts).
- [ ] **Cache reuse timing — must be same batch** — Cache only helps if similar requests batch together at the right time. Eviction matters.
- [ ] **TTFT vs TPOT vs throughput** — Time to First Token (user experience). Time Per Output Token. Tokens/sec (cost efficiency).
- [ ] **Tensor parallelism in vLLM** — --tensor-parallel-size flag. What it splits (attention heads + weights). Communication overhead.
- [ ] **vLLM V1 engine** — What changed from V0. Improved scheduler. Chunked prefill. AMD ROCm team enabled it.
- [ ] **benchmark_serving.py** — Know this tool exists. What flags it takes. What metrics it outputs.
- [ ] **vLLM on ROCm Docker** — docker pull rocm/vllm:latest. Prebuilt optimized image. VLLM_USE_TRITON_FLASH_ATTN=0 flag.

### Prefill vs Decode — CRITICAL DISTINCTION (6 topics)
- [ ] **What prefill does** — Processes entire input prompt in parallel. Compute-bound. Produces first KV cache. High GPU utilization.
- [ ] **What decode does** — Generates one token at a time. Memory-bandwidth bound, not compute bound. Much slower per token.
- [ ] **Why different bottlenecks** — Prefill is matrix multiplication (compute). Decode is KV cache reads (bandwidth). Same GPU, different limits.
- [ ] **TTFT dominated by prefill** — Long prompts = slow TTFT. Short prompts = fast TTFT. Decode affects TPOT, not TTFT.
- [ ] **Chunked prefill** — Split large prefill into chunks. Interleave with decode. Reduces TTFT variance. vLLM V1 uses this.
- [ ] **Disaggregated prefill/decode concept** — Mooncake/Distserve: separate GPUs for prefill and decode. Eliminates interference. Awareness only.

### KV Cache Deep Understanding (6 topics)
- [ ] **What K and V are** — Key and Value matrices from attention. Cached to avoid recomputation during decoding.
- [ ] **Why cache grows with sequence length** — Each token attends to all previous. Cache stores all previous K,V. Long context = memory problem.
- [ ] **Prefix sharing mechanics** — Identical prefixes → identical K,V → reusable. Requires requests batched together at right time.
- [ ] **Cache reuse requires timing alignment** — Cache hit only if cached prefix still in GPU memory when new request arrives. Eviction policy matters.
- [ ] **KV cache eviction** — When GPU memory full, older cache entries evicted. LRU or similar. Recomputation cost if needed again.
- [ ] **Cache limitations** — Only exact prefix matches help in vLLM. Partial overlap does not reuse. RadixAttention in SGLang handles partial.

### System Behavior Under Load (6 topics)
- [ ] **System saturation point** — Concurrency level where throughput stops growing but latency keeps rising. Your optimal operating point.
- [ ] **Queue buildup under overload** — When requests arrive faster than served, queue grows. Latency spikes. System feels stuck.
- [ ] **Little's Law intuition** — L = λW. Queue length = arrival rate × wait time. More concurrency at saturation = much higher latency.
- [ ] **Throughput vs latency tradeoff** — Larger batches improve throughput but increase latency. This is fundamental, not a bug.
- [ ] **GPU memory pressure behavior** — As batch size grows, KV cache competes with weights for memory. Preemption possible. Throughput collapses.
- [ ] **CPU vs GPU inference behavior** — CPU: high latency, no batching benefit, good for testing. GPU: fast ops, bandwidth bound in decode, needs warmup.

### Real-World Request Variability (5 topics)
- [ ] **Input length variability** — Real traffic has mixed prompt lengths. Short batch efficiently. Long cause memory pressure.
- [ ] **Output length unpredictability** — LLM output length unknown in advance. vLLM allocates speculatively. Too conservative = waste, aggressive = OOM.
- [ ] **Bursty arrival patterns** — Real traffic not uniform. Bursts cause queue buildup. Your batching window absorbs small bursts. Large bursts still spike latency.
- [ ] **Shared vs unique prefix distribution** — Many requests share system prompt (shared prefix). Suffixes unique. Prefix caching helps shared part only.
- [ ] **Why synthetic benchmarks differ from production** — Fixed lengths, uniform arrival, same prompts. Production is messier. Always caveat results.

### Your vLLM Project (6 topics)
- [ ] **Architecture cold — no hesitation** — FastAPI → queue → scheduler (50ms window) → prefix grouping → vLLM. Draw it on paper.
- [ ] **Why external scheduler on top of vLLM** — vLLM batches reactively. Your scheduler shapes input proactively. Different layers, different problems.
- [ ] **Why 50ms batching window** — Shorter = lower latency, worse batching. Longer = better batching, higher latency. 50ms is reasonable tradeoff.
- [ ] **Actual numbers you measured** — Even CPU numbers count. Latency with/without scheduler. Batch sizes formed. P50/P95 latencies.
- [ ] **Limitations of your project** — Single node, Python only, no cross-GPU KV transfer. Be honest. Shows self-awareness.
- [ ] **What you'd build next** — Real GPU experiments. Metrics dashboard. Tensor parallelism. Failure handling and timeouts.

---

## 🟡 SHOULD KNOW (40 topics) — Will Probably Come Up

### SGLang (4 topics)
- [ ] **RadixAttention** — Core innovation. Radix tree for automatic KV cache reuse across partial prefix matches. Handles what vLLM cannot.
- [ ] **SGLang vs vLLM tradeoffs** — SGLang better for complex multi-call programs and agents. vLLM better for simple single-call serving.
- [ ] **Structured output / constrained decoding** — SGLang's strength. JSON mode, regex constraints. Important for agentic systems.
- [ ] **SGLang runtime architecture** — Detokenizer, scheduler, model executor. How it differs from vLLM's architecture.

### AMD + ROCm Awareness (5 topics)
- [ ] **MI300X is AMD's flagship** — 192GB HBM3 memory (more than H100's 80GB). What the team benchmarks on.
- [ ] **ROCm = AMD's CUDA** — Open source GPU compute stack. HIP = AMD's CUDA kernel language. Know what it is, don't code in it.
- [ ] **AMD's 3-path attention routing** — ROCM_AITER_FA dispatches prefill, extend, decode to separate optimized kernels.
- [ ] **AMD MAD repository** — ROCm/MAD on GitHub. Standard AMD benchmarking tool. Know it exists.
- [ ] **rocm-smi command** — Check GPU memory, utilization, temperature. AMD's nvidia-smi. You'll use this.

### RAG — Production Knowledge (7 topics)
- [ ] **Full pipeline components** — Loading → chunking → embedding → vector store → retrieval → reranking → generation.
- [ ] **Chunking strategies** — Fixed size, semantic, hierarchical. You did hierarchical with Docling. Know why each is used.
- [ ] **Embedding models** — Dense vectors. Model choice matters (domain, dimension, speed). BGE, E5, OpenAI ada.
- [ ] **Vector similarity search** — Cosine similarity. HNSW index for approximate nearest neighbor. Why exact search doesn't scale.
- [ ] **Reranking** — Why retrieval alone isn't enough. Cross-encoder rerankers. Latency vs quality tradeoff.
- [ ] **RAG failure modes** — Retrieval misses. Context stuffing. Hallucination. Chunking too coarse or fine.
- [ ] **Your bank project — honest framing** — POC, not fully deployed. GPU procurement delays. <10s latency in dev. Be honest.

### MCP — Model Context Protocol (5 topics)
- [ ] **What problem MCP solves** — Standardizes LLM connections to external tools. No custom integration per tool. Like USB-C for AI.
- [ ] **MCP architecture** — Client (your app) ↔ Server (wraps tool) ↔ actual tool. Standard JSON-RPC protocol.
- [ ] **Tools vs Resources vs Prompts** — Tools = callable functions. Resources = readable data. Prompts = reusable templates.
- [ ] **MCP + vLLM connection** — AMD ran workshop on vLLM + MCP for multi-agent systems. Why this role cares about MCP.
- [ ] **MCP on your resume — own it** — Explain what you built with it. Even basic usage more than most people have.

### Agentic AI Concepts (6 topics)
- [ ] **ReAct pattern** — Reasoning + Acting loop. Think → Act → Observe → repeat. Foundation of agents.
- [ ] **Tool use / function calling** — How LLMs call external functions. JSON schema for tool definitions.
- [ ] **Multi-agent orchestration** — Orchestrator routes to specialist agents. Your Beacon project does this.
- [ ] **Agent memory types** — In-context (short term). External vector store (long term). When each is used.
- [ ] **Google A2A + ADK** — Used in Beacon. Agent-to-Agent secure handoffs. Honest: basic usage, worked around limitations.
- [ ] **LangGraph conceptual** — Stateful workflows as graphs. Nodes = actions. Edges = transitions. Agentic extension of LangChain.

### Benchmarking Methodology (5 topics)
- [ ] **Fair model comparison** — Same hardware, same prompts, same concurrency, same output length. Control all variables.
- [ ] **Concurrency sweep pattern** — Increase concurrency until throughput saturates. Find the knee. That's optimal point.
- [ ] **Warmup before benchmarking** — GPU kernels JIT compile first run. Discard first N results. AMD docs mention this.
- [ ] **Why P99 matters** — Users experience tail latency, not average. Good P50 + bad P99 = feels broken in production.
- [ ] **Input/output length distribution** — Fixed lengths unrealistic. Random distribution with controlled mean is better.

### Transformer Fundamentals (3 topics)
- [ ] **Self-attention mechanism** — Q, K, V matrices. Attention = softmax(QK^T / sqrt(d_k)) × V.
- [ ] **Why KV cache exists** — Autoregressive decoding recomputes attention each step. Cache avoids recomputation. Without it, O(n²).
- [ ] **MoE basics** — Sparse model. Subset of experts activated per token. DeepSeek uses this. Conceptual awareness.

### Git and Dev Workflow (3 topics)
- [ ] **Core commands cold** — clone, add, commit, push, pull, branch, checkout, merge, rebase, stash.
- [ ] **PR workflow** — Branch → commit → push → PR → review → merge. Standard team process.
- [ ] **Your GitHub looks professional** — Clean README, architecture diagram, setup instructions, results table. This is what they see first.

### Soft Skills (3 topics)
- [ ] **Your 90-second story** — VIT → Accenture (RAG + agents) → curious about inference → built vLLM → want to go deeper. Practice out loud.
- [ ] **Honest gap framing** — "I haven't worked with X but I understand what problem it solves." Confident, not apologetic.
- [ ] **Three questions to ask them** — Day-to-day workflow. Biggest ROCm vs CUDA gap. What they recommend you learn first.

### Failure Handling Basics (5 topics)
- [ ] **What happens when vLLM OOMs** — Out of memory error. Request dropped or preempted. vLLM has preemption strategies (swap to CPU, recompute).
- [ ] **Request timeout handling** — Long-running requests need timeouts. Client gets 504. Queue must drop stale requests.
- [ ] **Model loading failures** — Wrong path, insufficient memory, ROCm driver issues. Debug with logs, rocm-smi, check model size.
- [ ] **Graceful degradation** — Under extreme load, reject new requests (503) rather than let queue grow infinitely. Backpressure.
- [ ] **rocm-smi basics** — AMD's nvidia-smi. Check GPU memory, utilization, temperature. You'll use this when benchmarking.

---

## ⚫ SKIP ENTIRELY (These Won't Help)

- [ ] **DSA and sorting algorithms** — Not relevant for this role.
- [ ] **C++ / HIP kernel development** — Too deep. Awareness only.
- [ ] **RIXL / MORI / DeepEP internals** — Senior-level infra. Know what they solve, not how.
- [ ] **SLURM cluster management** — Job scheduler. Know it exists. Nothing more.
- [ ] **OOP design patterns** — Won't come up in this context.
- [ ] **Azure certification details** — You crammed it. That's fine and honest.

---

## 📊 Study Strategy by Timeline

### WEEK 1 (Foundations)
Priority: MUST KNOW sections in order:
1. vLLM Core Engine (10 topics) — 3 hours
2. Prefill vs Decode (6 topics) — 2 hours
3. KV Cache (6 topics) — 2 hours
4. Your vLLM Project (6 topics) — 3 hours

**Total: ~10 hours. Goal: Understand the concepts, know your own project cold.**

### WEEK 2 (GPU Experiments)
When you get AMD GPU access:
1. System Behavior Under Load (6 topics) — 1 hour (while running experiments)
2. Real-World Variability (5 topics) — 1 hour
3. AMD + ROCm (5 topics) — 1 hour
4. Run the 4 benchmark experiments — 4-5 hours

**Total: ~8 hours. Goal: Real numbers on real hardware, deep understanding from observation.**

### WEEK 3 (Interview Prep)
1. SHOULD KNOW sections — 5 hours (skim, don't deep dive)
2. Practice your 90-second story — 1 hour
3. Soft skills and questions to ask — 1 hour
4. One final review of MUST KNOW — 1 hour

**Total: ~8 hours. Goal: Confidence, not perfection. You should know the story you're telling.**

---

## 🎯 Benchmark Experiments Checklist

### Experiment 1: Model Comparison
- [ ] Spin up Qwen-7B, Llama-8B, Qwen-14B on AMD GPU
- [ ] Run each with TP=1, cache OFF, concurrency=10
- [ ] Measure: TTFT, P95 latency, tokens/sec, GPU memory
- [ ] Record in results table

### Experiment 2: Concurrency Sweep
- [ ] Use Qwen-7B only
- [ ] Run with concurrency: 1 → 5 → 10 → 20
- [ ] Measure: throughput and P95 latency at each level
- [ ] Find the saturation point (where throughput stops growing)
- [ ] Record all numbers

### Experiment 3: Scheduler vs No Scheduler
- [ ] Setup A: requests → vLLM directly (baseline)
- [ ] Setup B: requests → your queue → scheduler → vLLM
- [ ] Same 20 concurrent requests on Qwen-7B
- [ ] Measure: batch sizes formed, P95 latency, throughput
- [ ] Compare results

### Experiment 4: Prefix Cache (Positive + Negative Control)
- [ ] **Positive**: 20 requests, shared 500-token prefix, cache ON vs OFF
- [ ] **Negative**: 20 requests, random unique prompts, cache ON vs OFF
- [ ] Measure: TTFT difference in each case
- [ ] Record findings

---

## 🎤 Interview Day Checklist

- [ ] Know your 90-second story — practice it out loud 5 times
- [ ] Have your GitHub project link ready
- [ ] Have your results table screenshot ready
- [ ] Know what rocm-smi command outputs
- [ ] Know what vLLM continuous batching actually does
- [ ] Know when prefix caching helps vs doesn't help
- [ ] Know your project's limitations — be honest about them
- [ ] Have three questions ready to ask them
- [ ] Show your experiments — talk through the numbers
- [ ] Be confident about what you know, honest about what you don't

---

## Final Reminders

**You do not need to:**
- Know C++ or HIP
- Understand RIXL/MORI internals
- Be an expert in anything
- Have worked on production GPUs before

**You do need to:**
- Understand prefill vs decode (most people don't)
- Know when prefix caching helps (most people handwave this)
- Have real numbers from your GPU experiments
- Be able to explain your own project cold
- Be honest about gaps while showing hunger to learn

**The interview is not a test.** He already knows your background. He's checking:
1. Are you sharp enough to ramp on inference systems?
2. Can you communicate clearly about technical concepts?
3. Do you have genuine curiosity about the problem space?

All three are answered by your resume, your project, and how you talk about it.

---

**Last update:** April 2026  
**Total prep time:** ~26 hours over 3 weeks  
**This is enough.** Don't over-prepare. Focus on understanding, not memorization.
