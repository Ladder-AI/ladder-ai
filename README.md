# Ladder + TTFT

Custom Ladder implementation on any Complex problem for LLM. a reimplementation of the paper “LADDER: SELF-IMPROVING LLMS THROUGH RECURSIVE PROBLEM DECOMPOSITION”
https://arxiv.org/pdf/2503.00735

![workflow](./assets/workflow_version2.svg)

## Finetuned Qwen2-0.5B with Ladder (Model Response)

![Ladder-Finetuned](./assets/finetuned-ladder-answer.png)

# setup

## install from source using PDM

```bash
git clone git@github.com:AbdelrahmanAbounida/ladder.git
cd ladder
pdm install
```

# Run

## our main usecase (Graph problem)

```bash
python src/main.py
```

## TODO

### Dataset Generation

- [x] LLM Intelligence ratio Equation
- [x] Custom Verification Method if required (for our Graph Usecase)
- [ ] DatasetGenerator > Generate subproblems according to the model intelligence ratio (step3)
- [ ] Difficulty Engine should decide the level of difficulty to be generated and what transformations to be applied
- [ ] Verification engine should use the small llm to be tuned not the Larger one
- [ ] LLM Engine (temperature cycling and persona based prompts for different operations like variant generation)

### Ladder

- [x] Ladder Finetuning Process
- [x] GRPO Implementation
- [x] reward functions

### TTRL

- [ ] TTRL Implementation
- [ ] Data Generation in a loop

### Others

- [ ] General Configurations for all Constants and Hyper Parameters
- [ ] implement different interfaced for different models to be used (HF, Ollama, VLLM, deepspeed, LiteLLM,..)
- [ ] LLMS Benchmarking
- [ ] Metrics and other evaluation methods
- [ ] implement more usecases if required for diverse benchmarking
- [ ] use accelerate / PEFT / deepspeed and vllm to speed up training process

### Production

- [ ] Documentation
- [x] packaging
- [x] CICD
- [ ] Testing
