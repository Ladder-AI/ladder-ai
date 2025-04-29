# Ladder + TTFT

Custom Ladder implementation on any Complex problem for LLM. a reimplementation of the paper “LADDER: SELF-IMPROVING LLMS THROUGH RECURSIVE PROBLEM DECOMPOSITION”
https://arxiv.org/pdf/2503.00735

![workflow](./assets/workflow_version2.svg)

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

- [ ] LLM Intelligence ratio Equation
- [ ] Generate subproblems according to the model intelligence ratio
- [ ] Difficulty Engine should decide the level of difficulty to be generated and what transformations to be applied
- [ ] Verification engine should use the small llm to be tuned not the Larger one

### Ladder

- [ ] Ladder Finetuning Process
- [ ] GRPO Implementation

### TTRL

- [ ] TTRL Implementation
- [ ] Data Generation in a loop

### Others

- [ ] implement different interfaced for different models to be used (HF, Ollama,..)
- [ ] LLMS Benchmarking
- [ ] Metrics and other evaluation methods
