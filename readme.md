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
- [Generative AI](#generative-ai)
  - [LLM Tools](#llm-tools)
  - [Agent Frameworks](#agent-frameworks)
  - [Development Tools](#development-tools)
- [Livebooks & Examples](#livebooks--examples)
- [Resources](#resources)

## Core Tools

- [Nx](https://github.com/elixir-nx/nx) - Tensors for Elixir with compilation to CPU/GPU. It is the base for a lot of other libraries.
- [Explorer](https://github.com/elixir-explorer/explorer) - Series and dataframes for data exploration in Elixir.
- [Livebook](https://livebook.dev/) - Write interactive and collaborative notebooks, with integrations to databases, messaging, visualization and more.
- [Kino](https://github.com/livebook-dev/kino) - Render rich and interactive output. Used in Livebook.

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

### Deep Learning

- [Axon](https://github.com/elixir-nx/axon) - Neural Networks for Elixir. Built with Nx.
- [Bumblebee](https://github.com/elixir-nx/bumblebee) - Pre-trained neural network models on top of Axon. Provides integration with [Hugging Face](https://huggingface.co/).
- [Ortex](https://github.com/elixir-nx/ortex) - Wrapper around ONNX. Enables you to run ONNX models using Nx.

### Computer Vision

- [Evision](https://github.com/cocoa-xu/evision) - OpenCV bindings for Elixir/Erlang.
- [NxImage](https://github.com/elixir-nx/nx_image) - Image processing in Nx.

## Generative AI

### LLM Tools
- [Honeycomb](https://github.com/seanmor5/honeycomb) - Fast LLM inference service and library built on Elixir, Bumblebee, and EXLA with OpenAI API compatibility.
- [Instructor.ex](https://github.com/thmsmlr/instructor_ex) - Structured outputs from LLMs using Ecto schemas. Works with OpenAI, llama.cpp and Bumblebee.
- [Mentor](https://github.com/zoedsoupe/mentor) - Library for generating validated structured outputs from LLMs with automatic retries and schema validation.
- [Ollama-ex](https://github.com/lebrunel/ollama-ex) - Elixir client for Ollama API with support for completions, chat, tools, and function calling.
- [Rag](https://github.com/bitcrowd/rag) - Library for building Retrieval Augmented Generation (RAG) systems with support for vector stores like pgvector and chroma.
- [TextChunker](https://github.com/revelrylabs/text_chunker_ex) - Semantic text chunking library optimized for vector embedding and RAG applications.

### Agent Frameworks
- [Jido](https://github.com/agentjido/jido) - Framework for building autonomous, distributed agent systems with modular actions, stateful agents, and sensors. AI-framework agnostic.
- [LangChain](https://github.com/brainlid/langchain) - Framework for developing applications powered by language models, with support for OpenAI, Anthropic, Google, and Bumblebee models.
- [SwarmEx](https://github.com/nrrso/swarm_ex) - Lightweight library for AI agent orchestration with built-in telemetry and tool integration.

### Development Tools
- [HexDocs MCP](https://github.com/bradleygolden/hexdocs-mcp) - Enables semantic search of Elixir package documentation for AI assistants via Model Context Protocol (MCP).
- [Hermes MCP](https://github.com/cloudwalk/hermes-mcp) - SDK for the Model Context Protocol (MCP) with support for multiple transport options (STDIO, HTTP/SSE, WebSocket).
- [MCP Proxy](https://github.com/tidewave-ai/mcp_proxy_elixir) - Proxy that connects STDIO-based MCP clients to HTTP-based Server-Sent Events (SSE) MCP servers.


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

- [(2025) Keynote: Designing LLM Native systems - Sean Moriarity](https://www.youtube.com/watch?v=R9JRhIKQmqk)
- [(2024) Using LLMs and AI Agents to super power your Phoenix apps - Byron Saltysiak](https://www.youtube.com/watch?v=Hnpt2zv0rVw)
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

- (2023) [From Python to Elixir Machine Learning](https://www.thestackcanary.com/from-python-pytorch-to-elixir-nx/) - Nice wrapup on what you gain from the Elixir ecosystem for Machine Learning.

## Contributions

Contributions welcome! Read the [contribution guidelines](contributing.md) first.

## License

This project is licensed under the [CC0 License](LICENSE.md). Feel free to use, share, and adapt the content.
