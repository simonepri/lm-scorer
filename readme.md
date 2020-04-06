<h1 align="center">
  <b>lm-scorer</b>
</h1>
<p align="center">
  <!-- PyPi -->
  <a href="https://pypi.org/project/lm-scorer">
    <img src="https://img.shields.io/pypi/v/lm-scorer.svg" alt="PyPi version" />
  </a>
  <br />
  <!-- Lint -->
  <a href="https://github.com/simonepri/lm-scorer/actions?query=workflow:lint+branch:master">
    <img src="https://github.com/simonepri/lm-scorer/workflows/lint/badge.svg?branch=master" alt="Lint status" />
  </a>
  <!-- Test - macOS -->
  <a href="https://github.com/simonepri/lm-scorer/actions?query=workflow:test-macos+branch:master">
    <img src="https://github.com/simonepri/lm-scorer/workflows/test-macos/badge.svg?branch=master" alt="Test macOS status" />
  </a>
  <!-- Test - Ubuntu -->
  <a href="https://github.com/simonepri/lm-scorer/actions?query=workflow:test-ubuntu+branch:master">
    <img src="https://github.com/simonepri/lm-scorer/workflows/test-ubuntu/badge.svg?branch=master" alt="Test Ubuntu status" />
  </a>
  <br />
  <!-- Code style -->
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style" />
  </a>
  <!-- Linter -->
  <a href="https://github.com/PyCQA/pylint">
    <img src="https://img.shields.io/badge/linter-pylint-ce963f.svg" alt="Linter" />
  </a>
  <!-- Types checker -->
  <a href="https://github.com/PyCQA/pylint">
    <img src="https://img.shields.io/badge/types%20checker-mypy-296db2.svg" alt="Types checker" />
  </a>
  <!-- Test runner -->
  <a href="https://github.com/pytest-dev/pytest">
    <img src="https://img.shields.io/badge/test%20runner-pytest-449bd6.svg" alt="Test runner" />
  </a>
  <!-- Task runner -->
  <a href="https://github.com/illBeRoy/taskipy">
    <img src="https://img.shields.io/badge/task%20runner-taskipy-abe63e.svg" alt="Task runner" />
  </a>
  <!-- Build tool -->
  <a href="https://github.com/python-poetry/poetry">
    <img src="https://img.shields.io/badge/build%20system-poetry-4e5dc8.svg" alt="Build tool" />
  </a>
  <br />
  <!-- License -->
  <a href="https://github.com/simonepri/lm-scorer/tree/master/license">
    <img src="https://img.shields.io/github/license/simonepri/lm-scorer.svg" alt="Project license" />
  </a>
</p>
<p align="center">
  📃 Language Model based sentences scoring library
</p>

## Synopsis

This package provides a simple programming interface to score sentences using different ML [language models](wiki:language-model).

A simple [CLI](#cli) is also available for quick prototyping.

## Install

```bash
pip install lm-scorer
```

## Usage

```python
from lm_scorer.models.auto import AutoLMScorer as LMScorer

LMScorer.supported_model_names()
# => ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", distilgpt2"]

scorer = LMScorer.from_pretrained("gpt2")

scorer.score("I like this package.")
# => -25.835
scorer.score("I like this package.", return_tokens=True)
# => -25.835, {
#   "I": -3.9997,
#   "Ġlike": -5.0142,
#   "Ġthis": -2.5178,
#   "Ġpackage": -7.4062,
#   ".": -1.2812,
#   "<|endoftext|>": -5.6163,
# }

scorer.score("I like this package.", return_log_prob=False)
# => 6.0231e-12
scorer.score("I like this package.", return_log_prob=False, return_tokens=True)
# => 6.0231e-12, {
#   "I": 0.018321,
#   "Ġlike": 0.0066431,
#   "Ġthis": 0.080633,
#   "Ġpackage": 0.00060745,
#   ".": 0.27772,
#   "<|endoftext|>": 0.0036381,
# }

```

## CLI

<img src="https://github.com/simonepri/lm-scorer/raw/master/media/cli.gif" alt="lm-scorer cli" width="225" align="right"/>

The pip package includes a CLI that you can use to score sentences.

```
usage: lm-scorer [-h] [--model-name MODEL_NAME] [--tokens] [--log-prob]
                 [--debug]
                 sentences-file-path

Get sentences probability using a language model.

positional arguments:
  sentences-file-path   A file containing sentences to score, one per line. If
                        - is given as filename it reads from stdin instead.

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME, -m MODEL_NAME
                        The pretrained language model to use. Can be one of:
                        gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2.
  --tokens, -t          If provided it provides the probability of each token
                        of each sentence.
  --log-prob, -lp       If provided log probabilities are returned instead.
  --debug               If provided it provides additional logging in case of
                        errors.
```

## Authors

- **Simone Primarosa** - [simonepri][github:simonepri]

See also the list of [contributors][contributors] who participated in this project.


## License

This project is licensed under the MIT License - see the [license][license] file for details.



<!-- Links -->

[start]: https://github.com/simonepri/lm-scorer#start-of-content
[license]: https://github.com/simonepri/lm-scorer/tree/master/license
[contributors]: https://github.com/simonepri/lm-scorer/contributors

[wiki:language-model]: https://en.wikipedia.org/wiki/Language_model

[github:simonepri]: https://github.com/simonepri
