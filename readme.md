<div align="center">
  
![Awesome ML & GenAI in Elixir logo](https://github.com/user-attachments/assets/19fc503d-1123-4785-b5b2-570b0377c4ba)

# Awesome ML & GenAI in Elixir

A curated list of Machine Learning (ML) and Generative AI (GenAI) packages and resources for the [Elixir](https://elixir-lang.org/) programming language.

Besides giving an overview for experienced Elixir developers, this list can be useful for ML and AI practitioners looking for other ecosystems.

</div>
<br />

## Contents

- [Core Tools](#core-tools)
- [Machine Learning](#machine-learning)
  - [Traditional Machine Learning](#traditional-machine-learning)
  - [Deep Learning](#deep-learning)
  - [Computer Vision](#computer-vision)
  - [Vector Search & Similarity](#vector-search--similarity)
- [Generative AI](#generative-ai)
  - [LLM Tools](#llm-tools)
  - [Agent Frameworks](#agent-frameworks)
  - [Development Tools](#development-tools)
- [Livebooks & Examples](#livebooks--examples)
- [Resources](#resources)
  - [Books](#books)
  - [Videos](#videos)
  - [Articles](#articles)
  - [Discussions](#discussions)

## Core Tools

- [Nx](https://github.com/elixir-nx/nx) - Tensors for Elixir with compilation to CPU/GPU. It is the base for a lot of other libraries.
- [Explorer](https://github.com/elixir-explorer/explorer) - Series and dataframes for data exploration in Elixir.
- [Livebook](https://livebook.dev/) - Write interactive and collaborative notebooks, with integrations to databases, messaging, visualization and more.
- [Kino](https://github.com/livebook-dev/kino) - Render rich and interactive output. Used in Livebook.
- [Pythonx](https://github.com/livebook-dev/pythonx) - Embeds a Python interpreter directly into Elixir via NIF, running in the same OS process as the BEAM. Enables Elixir apps and Livebooks to call Python ML libraries directly.
- [EMLX](https://github.com/elixir-nx/emlx) - Nx backend and compiler for Apple MLX. Runs Nx, Axon and Bumblebee on Apple Silicon GPUs.
- [Emily](https://github.com/ausimian/emily) - Alternative MLX-based Nx backend with fused transformer kernels, running Nx.Serving and Bumblebee on Metal.
- [Handoff](https://github.com/polvalente/handoff) - Nx-compatible library for building and executing distributed DAGs of dependent functions across BEAM nodes, with per-node cost and allocation requirements.
- [NxSignal](https://github.com/elixir-nx/nx_signal) - Digital signal processing on Nx: FFT, filters and spectrograms. The DSP layer Bumblebee's audio models build on.
- [Tokenizers](https://github.com/elixir-nx/tokenizers) - Bindings to Hugging Face Tokenizers. The tokenization layer used by Bumblebee.

## Machine Learning

### Traditional Machine Learning

- [Scholar](https://github.com/elixir-nx/scholar) - Traditional machine learning tools built on top of Nx. Implements algorithms for:
  - Classification
  - Regression
  - Clustering
  - Dimensionality reduction
  - Metrics and preprocessing
- [EXGBoost](https://github.com/acalejos/exgboost) - Decision Trees implemented using the [XGBoost C API](https://xgboost.readthedocs.io/en/latest/c.html).
- [Mockinjay](https://github.com/acalejos/mockingjay) - Implementation of Microsoft's [Hummingbird](https://github.com/microsoft/hummingbird) library for converting trained Decision Tree models into Nx tensor computations.
- [Soothsayer](https://github.com/georgeguimaraes/soothsayer) - Time series forecasting library inspired by Facebook's Prophet and NeuralProphet.
- [Ulam](https://github.com/tmbb/ulam_ex) - Elixir interface to [Stan](https://mc-stan.org/), a probabilist programming language.
- [eXMC](https://github.com/borodark/eXMC) - Probabilistic programming on the BEAM inspired by PyMC: declarative models, NUTS sampling and Bayesian diagnostics on Nx tensors.

### Deep Learning

- [Axon](https://github.com/elixir-nx/axon) - Neural Networks for Elixir. Built with Nx.
- [Bumblebee](https://github.com/elixir-nx/bumblebee) - Pre-trained neural network models on top of Axon. Provides integration with [Hugging Face](https://huggingface.co/).
- [Ortex](https://github.com/elixir-nx/ortex) - Wrapper around ONNX. Enables you to run ONNX models using Nx.
- [Edifice](https://github.com/blasphemetheus/edifice) - Catalog of ready-to-use neural network architectures for Nx/Axon, from MLPs and transformers to Mamba, GNNs and VAEs, behind a uniform build API.
- [LlamaCppEx](https://github.com/nyo16/llama_cpp_ex) - llama.cpp bindings for running GGUF models locally on Metal, CUDA, Vulkan or CPU, with Hugging Face Hub downloads, streaming and structured output.

### Computer Vision

- [Evision](https://github.com/cocoa-xu/evision) - OpenCV bindings for Elixir/Erlang.
- [NxImage](https://github.com/elixir-nx/nx_image) - Image processing in Nx.
- [YOLO](https://github.com/poeticoding/yolo_elixir) - Real-time object detection using YOLOv8 models with 38ms processing time and optional Rust NIF for performance.
- [ImageVision](https://github.com/elixir-image/image_vision) - Classification, detection, segmentation, background removal, captioning and zero-shot labels for `Image` structs, backed by Bumblebee.

### Vector Search & Similarity

- [hnswlib](https://github.com/elixir-nx/hnswlib) - Elixir binding for the hnswlib approximate nearest neighbour library. In-process vector search with precompiled NIFs.
- [Leidenfold](https://github.com/georgeguimaraes/leidenfold) - Elixir bindings for the Leiden community detection algorithm.
- [Qdrant](https://github.com/marinac-dev/qdrant) - Client for the Qdrant vector database REST API.
- [Stephen](https://github.com/georgeguimaraes/stephen) - ColBERT-style neural retrieval for Elixir.
- [Turbopuffer](https://github.com/jallum/turbopuffer) - Client for the Turbopuffer vector and BM25 full-text search API, with hybrid search.
- [Vettore](https://github.com/elchemista/vettore) - In-memory vector database on ETS with Rust-accelerated distance functions and HNSW indexing.

## Generative AI

### LLM Tools

- [Arcana](https://github.com/georgeguimaraes/arcana) - Embeddable RAG library for Elixir/Phoenix with agentic pipelines and dashboard.
- [Aludel](https://github.com/ccarvalho-eng/aludel) - Phoenix-native LLM evaluation workbench: prompt tests, model comparison, rubric-based LLM judges, red-team suites, ExUnit assertions and CI gates.
- [Anthropix](https://github.com/lebrunel/anthropix) - Anthropic Messages API client with streaming and tool use.
- [AshAi](https://github.com/ash-project/ash_ai) - Structured outputs, vectorization and tool calling for your Ash application with LangChain integration and MCP server capabilities.
- [ClaudeCode](https://github.com/guess/claude_code) - SDK for embedding Claude as an agentic AI in Elixir apps with tool calling and MCP integration.
- [LLM Composer](https://github.com/doofinder/llm_composer) - Multi-provider LLM library with routing, fallback, streaming, and cost tracking for OpenAI, Anthropic, Gemini, and more.
- [ReqLLM](https://github.com/agentjido/req_llm) - A Req-based package to call LLM APIs that standardizes the API calls and responses for LLM providers.
- [Chunx](https://github.com/preciz/chunx) - Text chunking with token, word, sentence and semantic strategies, ported from Chonkie.
- [ClaudeAgentSDK](https://github.com/nshkrdotcom/claude_agent_sdk) - Elixir SDK for the Claude Code CLI with streaming and session management.
- [DSPEx](https://github.com/nshkrdotcom/ds_ex) - DSPy port for Elixir with data-driven prompt optimization.
- [ExOpenAI](https://github.com/dvcrn/ex_openai) - OpenAI SDK generated from the API spec, with full typespecs, docs and streaming, so new endpoints land quickly.
- [extractous_ex](https://github.com/Valian/extractous_ex) - Text and metadata extraction from PDFs, Office documents and more via the Extractous Rust library. Document ingestion for RAG pipelines.
- [Gemini.ex](https://github.com/nshkrdotcom/gemini_ex) - Elixir client for Google Gemini LLM supporting both AI Studio and Vertex AI.
- [GenAI](https://github.com/noizu-labs-ml/genai) - Multi-provider generative AI client with chat threads, media jobs and plugin extensions. Supports Anthropic, Gemini, OpenAI, Groq and local llama.cpp.
- [Hallmark](https://github.com/georgeguimaraes/hallmark) - Hallucination detection using Vectara's HHEM model, running locally on Bumblebee.
- [Handwave](https://github.com/martosaur/handwave) - LLM-powered control flow for Elixir: conditional logic, text rewriting, and routing decisions via natural language rather than code.
- [Honeycomb](https://github.com/seanmor5/honeycomb) - Fast LLM inference service and library built on Elixir, Bumblebee, and EXLA with OpenAI API compatibility.
- [Instructor.ex](https://github.com/thmsmlr/instructor_ex) - Structured outputs from LLMs using Ecto schemas. Works with OpenAI, llama.cpp and Bumblebee.
- [just_bash](https://github.com/elixir-ai-tools/just_bash) - Pure-Elixir bash interpreter and virtual filesystem for sandboxing agent tool calls, built by Knock to run a production agent.
- [JsonRemedy](https://github.com/nshkrdotcom/json_remedy) - JSON repair library for fixing malformed LLM outputs.
- [InstructorLite](https://github.com/martosaur/instructor_lite) - Lightweight structured outputs for LLMs using JSON schemas with multi-provider support including OpenAI, Anthropic, and Gemini.
- [Mentor](https://github.com/zoedsoupe/mentor) - Library for generating validated structured outputs from LLMs with automatic retries and schema validation.
- [Mistral](https://github.com/rodloboz/mistral) - Open-source Elixir client for the Mistral AI API covering chat completions, function calling, embeddings, streaming, OCR, fine-tuning, and batch processing.
- [Ollama-ex](https://github.com/lebrunel/ollama-ex) - Elixir client for Ollama API with support for completions, chat, tools, and function calling.
- [OpenAI.ex](https://github.com/cyberchitta/openai_ex) - OpenAI API client with streaming, file uploads, and Azure OpenAI support.
- [Rag](https://github.com/bitcrowd/rag) - Library for building Retrieval Augmented Generation (RAG) systems with support for vector stores like pgvector and chroma.
- [TOON](https://github.com/kentaro/toon_ex) - Token-Oriented Object Notation encoder and decoder for compact LLM prompts.
- [TextChunker](https://github.com/revelrylabs/text_chunker_ex) - Semantic text chunking library optimized for vector embedding and RAG applications.
- [xberg](https://github.com/xberg-io/xberg) - Document intelligence for RAG ingestion (parsing, OCR, tables, chunking across 100+ formats) with a Rust core and first-party Elixir bindings.
- [Tribunal](https://github.com/georgeguimaraes/tribunal) - LLM evaluation framework that provides tools for evaluating and testing LLM outputs, detecting hallucinations, and measuring response quality

### Agent Frameworks

- [A2A](https://github.com/actioncard/a2a-elixir) - Elixir implementation of the Agent-to-Agent (A2A) protocol with GenServer-based agents, JSON-RPC 2.0 dispatch, SSE streaming, skill-based discovery, and fleet supervision.
- [A2UI](https://github.com/actioncard/a2ui-elixir) - Phoenix LiveView renderer for the Agent-to-UI (A2UI) protocol, converting declarative JSONL agent messages into native LiveView components with two-way data binding.
- [Bazaar](https://github.com/georgeguimaraes/bazaar) - Elixir SDK for serving AI agent commerce protocols (UCP and ACP) from a single Phoenix handler. Supports Google Shopping agents (UCP) and OpenAI/Stripe agents (ACP) with automatic request/response translation between protocols.
- [A2A](https://github.com/actioncard/a2a-elixir) - Agent-to-Agent protocol server and client with GenServer-backed agents, multi-turn tasks, streaming and official TCK compliance.
- [Alloy](https://github.com/alloy-ex/alloy) - Minimal, OTP-native completion and tool-call loop. Provider-agnostic, with cost guards, and deliberately not a framework.
- [BeamWeaver](https://github.com/caudena/beam_weaver) - LangChain, LangGraph and Deep Agents patterns as native OTP workflows, with tools, memory, human-in-the-loop and tracing.
- [Jido](https://github.com/agentjido/jido) - Framework for building autonomous, distributed agent systems with modular actions, stateful agents, and sensors. AI-framework agnostic.
- [Jido.AI](https://github.com/agentjido/jido_ai) - LLM integration layer for Jido. Provides actions and reasoning strategies (ReAct, Chain-of-Thought, Tree-of-Thoughts) for building intelligent agents with OpenAI, Anthropic, and other providers.
- [LangEx](https://github.com/surgeventures/lang_ex) - LangGraph-style graph orchestration for durable, human-in-the-loop LLM agents.
- [Legion](https://github.com/software-mansion-labs/legion) - Runtime for AI agents that live inside your Elixir app and act by writing code, run in monitored sandboxes, with tools defined as Elixir modules. By Software Mansion.
- [LangChain](https://github.com/brainlid/langchain) - Framework for developing applications powered by language models, with support for OpenAI, Anthropic, Google, and Bumblebee models.
- [Sagents](https://github.com/sagents-ai/sagents) - Framework for interactive AI agents with OTP supervision, middleware composition, human-in-the-loop approvals, sub-agent delegation, and a Phoenix LiveView debugger.
- [SwarmEx](https://github.com/nrrso/swarm_ex) - Lightweight library for AI agent orchestration with built-in telemetry and tool integration.
- [Synapse](https://github.com/nshkrdotcom/synapse) - Multi-agent orchestration framework with Postgres persistence.

### Development Tools

- [AgentObs](https://github.com/lostbean/agent_obs) - LLM agent observability with telemetry, token tracking, and OpenTelemetry spans following OpenInference conventions.
- [Alike](https://github.com/georgeguimaraes/alike) - Semantic similarity testing library using a wave operator (`<~>`) for assertions. Tests whether sentences convey the same meaning rather than exact matches, ideal for validating LLM outputs.
- [Beamlens](https://github.com/beamlens/beamlens) - AI-powered runtime intelligence for the BEAM. Lives in your supervision tree and uses LLMs to explain metrics, diagnose incidents, detect anomalies, and trace message queue bottlenecks.
- [claude-code-elixir](https://github.com/georgeguimaraes/claude-code-elixir) - Collection of Claude Code plugins for Elixir development. Includes LSP integration, formatting and compilation hooks, and thinking skills for Elixir, Phoenix, Ecto, and OTP patterns.
- [Elixir Agent Tools](https://github.com/georgeguimaraes/elixir-agent-tools) - Elixir development skills for coding agents (language idioms, Phoenix, Ecto, OTP, Oban), with optional Mix checks and Expert language server integration. Works with Claude Code, Codex and OpenCode.
- [Evals](https://github.com/ash-project/evals) - Tool for evaluating AI language models on Elixir code generation with side-by-side model comparisons and automated testing.
- [llm_db](https://github.com/agentjido/llm_db) - LLM model metadata database with O(1) lookups for provider capabilities, pricing, and context limits. Packaged as a dependency snapshot with no runtime network calls needed.
- [LlmGuard](https://github.com/North-Shore-AI/LlmGuard) - AI firewall with prompt injection detection, PII redaction, and jailbreak prevention for LLM applications.
- [ExSlop](https://github.com/elixir-vibe/ex_slop) - Credo checks that catch AI-generated code slop: blanket rescues, narrator docs, N+1 queries and more.
- [HexDocs MCP](https://github.com/bradleygolden/hexdocs-mcp) - Enables semantic search of Elixir package documentation for AI assistants via Model Context Protocol (MCP).
- [Anubis MCP](https://github.com/zoedsoupe/anubis-mcp) - SDK for the Model Context Protocol (MCP) with support for multiple transport options (STDIO, HTTP/SSE, WebSocket).
- [ex_mcp](https://github.com/azmaveth/ex_mcp) - Complete Elixir implementation of the Model Context Protocol (v2025-11-25) with client and server support, multiple transports including native BEAM, and 2600+ tests.
- [Phantom MCP](https://github.com/dbernheisel/phantom_mcp) - MCP server framework for Plug with Streamable HTTP and stdio transports.
- [Symphony](https://github.com/openai/symphony) - OpenAI's reference implementation, written in Elixir, of an orchestrator that runs Codex coding agents against an issue tracker in isolated per-issue workspaces.
- [Tidewave Phoenix](https://github.com/tidewave-ai/tidewave_phoenix) - AI-powered development assistant for Phoenix web applications that connects editor AI assistants to web framework runtime via MCP.
- [Usage Rules](https://github.com/ash-project/usage_rules) - Tool for synchronizing LLM rules files with dependencies to prevent AI hallucinations and ensure consistent usage patterns.

## Livebooks & Examples

- [José Valim's Livebooks](https://github.com/josevalim/livebooks) - Livebooks that José used for talks and Advent of Code.
- [Programming Machine Learning](https://github.com/nickgnd/programming-machine-learning-livebooks) - Livebook notebooks with code examples for the [Programming Machine Learning book by Paolo Perrotta](https://pragprog.com/titles/pplearn/programming-machine-learning/)
- [Machine Learning in Elixir](https://github.com/charlieroth/machine-learning-in-elixir) - Livebooks following along with the book [Machine Learning in Elixir by Sean Moriarity](https://pragprog.com/titles/smelixir/machine-learning-in-elixir/)
- [Asynchronous Processing in Elixir](https://github.com/whatyouhide/guide_async_processing_in_elixir) - Interactive guide using Livebook to asynchronous data processing in Elixir.

## Resources

### Books

- [Machine Learning in Elixir - Learning to Learn with Nx and Axon (by Sean Moriarity)](https://pragprog.com/titles/smelixir/machine-learning-in-elixir/)
- [Genetic Algorithms in Elixir - Solve Problems Using Evolution (by Sean Moriarity)](https://pragprog.com/titles/smgaelixir/genetic-algorithms-in-elixir/)

### Videos

- [(2026) Keynote: Exoskeletons, not Autopilots - Zach Daniel](https://www.youtube.com/watch?v=7smgEJkgqO8)
- [(2026) AI-Powered Search at Scale - Jeff Weiss](https://www.youtube.com/watch?v=-a4A33hDtzA)
- [(2026) Building Real-Time AI Agents with Elixir - Kimutai Kiprotich](https://www.youtube.com/watch?v=XSXk0YE2New)
- [(2026) What if you could build production-ready RAG entirely in Elixir? - George Guimarães](https://www.youtube.com/watch?v=tNBQQSeuAkE)
- [(2026) Arbor: AI Agent Orchestration on the BEAM - Hysun Chung](https://www.youtube.com/watch?v=WoBoBaUpGo4)
- [(2025) Structured Generation and Logits Processing with Elixir - Chris Beck](https://www.youtube.com/watch?v=6r2MO2zk52A)
- [(2025) Detecting Confident Nonsense: Testing LLM-Driven Apps - Hernan Rivas Acosta](https://www.youtube.com/watch?v=qWbN_iW6F_c)
- [(2025) Concurrent AI Evaluation: Scaling Model Performance Monitoring with OTP - Ramiro Matteoda](https://www.youtube.com/watch?v=-TAevFJcQ4Q)
- [(2025) Beyond GenServers: Declarative AI Flows with gen_statem - Coby Benveniste](https://www.youtube.com/watch?v=duEZxrsTUA4)
- [(2025) MoodBot: Raising a Tiny Robot with Elixir, Nerves, and AI - Carsten Rösnick-Neugebauer](https://www.youtube.com/watch?v=7Z18Z99reTM)
- [(2025) Teaching the BEAM to See - Alvise Susmel](https://www.youtube.com/watch?v=zvFyCl4XEZI)
- [(2025) Keynote: A Survival Guide for the AI Age - Josh Price](https://www.youtube.com/watch?v=C7BWgWdhZwE)
- [(2025) Smarter Apps with Ash and the Model Context Protocol - Josh Price](https://www.youtube.com/watch?v=i1wVs7bgICU)
- [(2025) Nerves of Vision - Alvise Susmel](https://www.youtube.com/watch?v=LQHDaqvM2ik)
- [(2025) Who is doing the thinking? - Bruce Tate](https://www.youtube.com/watch?v=2MP1m1jZVJ0)
- [(2025) AI Agents - Elixir is all you need - Daniel Kentfield](https://www.youtube.com/watch?v=Wu3jXi1IyeM)
- [(2025) You can build an AI agent - Dennis Beatty](https://www.youtube.com/watch?v=Xg1Hib517c0)
- [(2025) Hey Ash, Let's Chat! Building Conversational AI with Ash AI - Jinkyou Son](https://www.youtube.com/watch?v=g9rWEbyHTRo)
- [(2025) Vibe Coding with Elixir: Harnessing AI to Build Real-World Apps - Micah Woods](https://www.youtube.com/watch?v=XPcC1MCRXMc)
- [(2025) Handoff: a new Nx-compatible library for distributed DAGs - Paulo Valente](https://www.youtube.com/watch?v=IhW6issVai0)
- [(2025) Scaling Edge AI with Elixir and Nerves - Vittoria Bitton & Paulo Valente](https://www.youtube.com/watch?v=nRq5Gnf-n3c)
- [(2025) Code Intelligence and RAG in Elixir with rag.ex - Chris Beck](https://www.youtube.com/watch?v=RK65aR2M5eE)
- [(2025) Meta Programming in Elixir: Dynamic Function Calling with LLMs - Conor Sinclair](https://www.youtube.com/watch?v=q2iW1x0gNKA)
- [(2025) Ash AI Launch - Zach Daniel](https://www.youtube.com/watch?v=PSrzruaby1M)
- [(2025) Whisperer: An Elixir-based Multi-Agent Workflow Framework - Sola Aremu & Ridwan Otun](https://www.youtube.com/watch?v=NrzN--WYU5M)
- [(2025) Keynote: Elixir's AI Future - Chris McCord](https://www.youtube.com/watch?v=6fj2u6Vm42E)
- [(2025) Keynote: Designing LLM Native systems - Sean Moriarity](https://www.youtube.com/watch?v=R9JRhIKQmqk)
- [(2025) Full-Stack AI with Elixir - George Guimarães](https://www.youtube.com/watch?v=toAIdOVCHCw)
- [(2025) Keynote: Code Generators are Dead. Long Live Code Generators - Chris McCord](https://www.youtube.com/watch?v=ojL_VHc4gLk)
- [(2024) Ship it! A Roadmap for Putting Nx into Production - Christopher Grainger](https://www.youtube.com/watch?v=5FlZHkc4Mq4)
- [(2024) Using LLMs and AI Agents to super power your Phoenix apps - Byron Saltysiak](https://www.youtube.com/watch?v=Hnpt2zv0rVw)
- [(2024) Soothsayer: Using NeuralProphet, Nx and Livebook to Forecast Business Data in Elixir - George Guimarães](https://www.youtube.com/watch?v=3LmKRrLUn5w)
- [(2023) A year in production with Machine Learning on the BEAM](https://www.youtube.com/watch?v=HP86Svk4hzI) (Explorer, Scholar, Bumblebee, Livebook)
- [(2023) Nx-powered decision trees](https://www.youtube.com/watch?v=rbmviKT6HkU) (Nx, EXGBoost)
- [(2023) Building AI apps with Elixir](https://www.youtube.com/watch?v=TfZI5-oQSqI)
- [(2023) MLOps in Elixir: Simplifying traditional MLOps with Elixir](https://www.youtube.com/watch?v=6aVnwj8WQq4) (Nx, Bumbleblee)
- [(2023) Fine-tuning language models with Axon](https://www.youtube.com/watch?v=-iZIZHgHa5M) (Axon)
- [(2023) Data wrangling with Livebook and Explorer](https://www.youtube.com/watch?v=U6nuPjyAUPw) (Livebook, Explorer)
- [(2022) The Future AI Stack by Chris Grainer](https://www.youtube.com/watch?v=Y2Nr4dNu6hI) (Explorer, Axon)
- [(2022) Announcing Bumblebee: pre-trained machine learning models for GPT2, StableDiffusion, and more](https://www.youtube.com/watch?v=g3oyh3g1AtQ) (Livebook, Bumblebee)
- [(2022) Axon: functional programming for deep learning](https://www.youtube.com/watch?v=NWXSiZ-vi-o) (Axon)

### Articles

- (2026) [Why Elixir is the best language for AI](https://dashbit.co/blog/why-elixir-best-language-for-ai) - José Valim makes a data-backed case for Elixir in the AI era: immutability, first-class docs, operational simplicity, and runtime introspection.
- (2026) [Your Agent Framework Is Just a Bad Clone of Elixir](https://georgeguimaraes.com/your-agent-orchestrator-is-just-a-bad-clone-of-elixir/) - George Guimarães argues that Python agent frameworks are independently rediscovering BEAM primitives, and Elixir already solves the hard parts of long-lived agent connections.
- (2026) [Distributed Python dataframes and machine learning with Livebook and Elixir](https://dashbit.co/blog/distributed-python-livebook) - José Valim runs Python across a cluster from Livebook: Pythonx with reproducible uv environments, zero-copy Arrow between Explorer and pandas, and Python workers on Fly or Kubernetes over Erlang distribution.
- (2026) Karlo Šmid's "Build an LLM from Scratch in Elixir" series, an Nx/Axon port of Sebastian Raschka's book: [Working with Text Data](https://karlosmid.com/2026/01/build-llm-from-scratch-chapter-2-working-with-text-data/), [The Attention Mechanism in Elixir](https://karlosmid.com/2026/03/coding-attention-mechanism-in-elixir/), [Implementing a GPT model from scratch](https://karlosmid.com/2026/05/implementing-a-GPT-model-from-scratch/), [Pretraining on unlabeled data](https://karlosmid.com/2026/05/pretraining-on-unlabeled-data-in-elixir/) and [Fine-tuning for classification](https://karlosmid.com/2026/06/fine-tuning-for-classification/), which fine-tunes GPT-2 for SMS spam detection in Nx.
- (2026) [Sherlock: from Hackathon to company-wide AI assistant](https://engineering.remote.com/blog/sherlock/) - Remote's engineering team builds a Phoenix, LiveView and Ash chat assistant that lets non-engineers query the codebase, streaming from OpenCode via Req.
- (2026) [Why I moved from Python to Elixir for AI work, and what I found on the other side](https://www.linkedin.com/pulse/why-i-moved-from-python-elixir-ai-work-what-found-usseglio-viretta-xgqjf/) - Alessandro Usseglio Viretta on preemptive scheduling versus async/await for long-running LLM calls.
- (2026) [BEAM Is a Suspiciously Good Fit for Agents](https://playground.tetraresearch.io/p/beam-is-a-suspiciously-good-fit-for) - Tyler O'Briant maps agent runtime requirements onto processes, supervision and message passing.
- (2026) [Elixir, Clojure, or Python for LLM Agents? Our Experience with All Three](https://sdtimes.com/programming-languages/elixir-clojure-or-python-for-llm-agents-our-experience-with-all-three/) - Artem Barmin compares building the same agent system in three languages and where each one breaks down.
- (2025) [Embedding Python in Elixir, it's Fine](https://dashbit.co/blog/running-python-in-elixir-its-fine) - Jonatan Kłosko introduces Pythonx: embedding Python directly in the BEAM via NIF with automatic virtual env management and same-process memory sharing.
- (2025) [Building a MCP Server in Elixir](https://hashrocket.com/blog/posts/building-a-mcp-server-in-elixir) - Hashrocket walks through building a real MCP server using Anubis, letting AI tools like Claude Code and Cursor interact directly with a Phoenix app.
- (2024) [Elixir and Machine Learning in 2024 so far](https://dashbit.co/blog/elixir-ml-s1-2024-mlir-arrow-instructor) - José Valim's mid-year ecosystem update covering Nx's move to MLIR, Apple Silicon support, Explorer's Arrow improvements, and structured outputs via instructor_ex.
- (2024) [What I mean when I say ML in Elixir is production-ready](https://cigrainger.com/elixirconf-eu-2024-keynote/) - Christopher Grainger makes the case for production ML on the BEAM: Nx.Serving for distributed batching, actor model for model supervision, and native integration with Phoenix, Oban, and Broadway.
- (2024) [AI GPU Clusters, From Your Laptop, With Livebook](https://fly.io/blog/ai-gpu-clusters-from-your-laptop-livebook/) - Chris McCord and José Valim demonstrate scaling to 64 GPU machines simultaneously from a local Livebook using FLAME and Nx's native BEAM clustering.
- (2024) [Training LoRA Models with Axon](https://dockyard.com/blog/2024/10/08/training-lora-models-with-axon) - Sean Moriarity's deep dive on fine-tuning LLMs in pure Elixir using LoRA and Axon's graph rewriting APIs.
- (2024) [Implementing Natural Conversational Agents with Elixir](https://seanmoriarity.com/2024/02/25/implementing-natural-conversational-agents-with-elixir/) - Sean Moriarity builds a voice AI assistant with Whisper, GPT-3.5, and ElevenLabs in Elixir, reducing latency from 4.5s to ~1s with Silero VAD and GPU acceleration.
- (2023) [From Python to Elixir Machine Learning](https://www.thestackcanary.com/from-python-pytorch-to-elixir-nx/) - Nice wrapup on what you gain from the Elixir ecosystem for Machine Learning.

### Discussions

- [State of developing agents with Elixir (not coding agents)](https://elixirforum.com/t/state-of-developing-agents-with-elixir-not-coding-agents/74313) - ElixirForum thread (2026) surveying the agent frameworks, gaps and production experiences in the ecosystem.
- [Elixir is the productivity language of the Agentic era](https://elixirforum.com/t/elixir-is-the-productivity-language-of-the-agentic-era/74544) - ElixirForum thread (2026) on why coding agents produce better Elixir than most languages.
- [OpenAI released a library that uses Elixir to orchestrate AI agents](https://elixirforum.com/t/openai-released-a-library-that-uses-elixir-to-orchestrate-ai-agents/74520) - ElixirForum discussion (2026) of Symphony and what it says about the BEAM for agent orchestration.
- [Nx ecosystem 0.13 library updates](https://elixirforum.com/t/nx-ecosystem-0-13-library-updates/76045) - Paulo Valente's release notes (2026) across Nx, EXLA, Torchx, EMLX and friends.
- [ElixirConf EU 2026: list of talks, slides and links](https://elixirforum.com/t/elixirconf-eu-2026-list-of-talks-slides-and-links/75144) - Community-maintained index of the conference's talks.

## Contributions

Contributions welcome! Read the [contribution guidelines](contributing.md) first.

## License

This project is licensed under the [CC0 License](LICENSE.md). Feel free to use, share, and adapt the content.
