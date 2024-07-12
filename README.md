# Tiny TextGrad

Like key-value, but with questions and answers.

[![PyPI version](https://badge.fury.io/py/tinytextgrad.svg)](https://badge.fury.io/py/tinytextgrad)
[![GitHub license](https://img.shields.io/github/license/witt3rd/tinytextgrad.svg)](https://github.com/witt3rd/tinytextgrade/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/witt3rd/tinytextgrad.svg)](https://github.com/witt3rd/tinytextgrad/issues)
[![GitHub stars](https://img.shields.io/github/stars/witt3rd/tinytextgrad.svg)](https://github.com/witt3rd/tinytextgrad/stargazers)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/dt_public.svg?style=social&label=Follow%20%40dt_public)](https://twitter.com/dt_public)

TinyTextGrad is an educational package inspired by the [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) paper, designed to provide hands-on understanding of how concepts from PyTorch and deep learning can be applied to generative AI, with a particular focus on automatic prompt optimization (APO).

This lightweight implementation applies backpropagation methods specifically tailored for text-based feedback, enabling effective optimization of the generation process for both generative and discriminative tasks. TinyTextGrad aims to demystify the inner workings of language model optimization and prompt engineering.

## Key Features

1. **Automatic Prompt Optimization (APO)**: Enhance reasoning capabilities by optimizing prompt phrases provided to language models (LLMs), improving accuracy in complex analytical tasks and decision-making processes.

2. **PyTorch-like Syntax**: Utilize familiar abstractions and syntax similar to PyTorch, making it easier for those with prior deep learning experience to adapt quickly.

3. **User-Friendly Design**: Accessible to end-users who may not have extensive background knowledge in machine learning or optimization algorithms.

4. **Educational Focus**: Built for learning and experimentation, allowing users to gain first-hand experience with concepts like automatic reverse mode differentiation, backpropagation, and gradient descent in the context of text generation.

## Installation

You can install TinyTextGrad using pip:

```bash
pip install tinytextgrad
```

## Usage

Here's a simple example of how to use TinyTextGrad for automatic prompt optimization:

```python
from tinytextgrad import optimize_prompt
from textwrap import dedent

initial_prompt = dedent("""
Analyze the given sentence and determine its primary emotion.
Respond with a single word: happy, sad, angry, or neutral.
""").strip()

inputs = [
    "I can't believe I won the lottery!",
    "The rain ruined our picnic plans.",
    "This traffic is making me late for work.",
    "The sky is cloudy today.",
    "She surprised me with tickets to my favorite band!",
    "I dropped my phone and cracked the screen.",
    "The customer service was incredibly rude.",
    "I'm going to the grocery store later.",
    "My best friend is moving away next month.",
    "The movie was exactly what I expected it to be.",
]

result = optimize_prompt(
    initial_prompt,
    "gpt-3.5-turbo",
    "gpt-4",
    inputs,
    num_iterations=3,
)

print("\n\nFinal optimized EMOTION_ANALYSIS_PROMPT:")
print(result.variable.value)

EMOTION_ANALYSIS_PROMPT = result
```

This example demonstrates how to use TinyTextGrad to optimize a prompt for emotion analysis. The `optimize_prompt` function takes an initial prompt, specifies the models to use for generation and evaluation, provides a list of input sentences, and performs optimization over a specified number of iterations.

The resulting optimized prompt can then be used for more accurate emotion analysis tasks.

## Requirements

- Python 3.7+
- Dependencies listed in `pyproject.toml`

## TODO

We're constantly working to improve TinyTextGrad. Here are some features and enhancements we're planning to implement:

- [ ] Automatic training set generation

  - Develop a system to automatically generate diverse and relevant training sets for prompt optimization.

- [ ] Momentum and gradient context

  - Implement momentum-based optimization techniques to potentially improve convergence speed and stability.
  - Introduce gradient context to better handle long-term dependencies in prompt optimization.

- [ ] Computation graph

  - Maintain a computation graph for multi-step operations, allowing for more complex optimizations.

- [ ] New loss functions
  - Develop and integrate additional loss functions tailored for specific tasks.
  - Allow users to easily define and use custom loss functions for their unique use cases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Donald Thompson - [@dt_public](https://twitter.com/dt_public) - <witt3rd@witt3rd.com>

Project Link: [https://github.com/witt3rd/tinytextgrad](https://github.com/witt3rd/tinytextgrad)
