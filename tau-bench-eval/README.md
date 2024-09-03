# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains

Paper: https://arxiv.org/abs/2406.12045

## Setup

1. Clone this repository:

```bash
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
ANYSCALE_API_KEY=...
```


## Run
Run a function calling agent on the τ-retail environment:

```bash
python run.py --env retail --model gpt-4o --max_concurrency 10
```

Set max concurrency according to your API limit.
