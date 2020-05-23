<h1 align="center">
  <b>lm-scorer</b>
</h1>
<p align="center">
  <!-- PyPi -->
  <a href="https://pypi.org/project/lm-scorer">
    <img src="https://img.shields.io/pypi/v/lm-scorer.svg" alt="PyPi version" />
  </a>
  <a href="https://colab.research.google.com/github/simonepri/lm-scorer/blob/master/examples/lm_scorer.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
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
You can run it locally or on directly on Colab using [this notebook][colab:lm-scorer].

Do you believe that this is *useful*?
Has it *saved you time*?
Or maybe you simply *like it*?  
If so, [support this work with a Star ⭐️][start].

## Install

```bash
pip install lm-scorer
```

## Usage

```python
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

# Available models
list(LMScorer.supported_model_names())
# => ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", distilgpt2"]

# Load model to cpu or cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)

# Return token probabilities (provide log=True to return log probabilities)
scorer.tokens_score("I like this package.")
# => (scores, ids, tokens)
# scores = [0.018321, 0.0066431, 0.080633, 0.00060745, 0.27772, 0.0036381]
# ids    = [40,       588,       428,      5301,       13,      50256]
# tokens = ["I",      "Ġlike",   "Ġthis",  "Ġpackage", ".",     "<|endoftext|>"]

# Compute sentence score as the product of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="prod")
# => 6.0231e-12

# Compute sentence score as the mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="mean")
# => 0.064593

# Compute sentence score as the geometric mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="gmean")
# => 0.013489

# Compute sentence score as the harmonic mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="hmean")
# => 0.0028008

# Get the log of the sentence score.
scorer.sentence_score("I like this package.", log=True)
# => -25.835

# Score multiple sentences.
scorer.sentence_score(["Sentence 1", "Sentence 2"])
# => [1.1508e-11, 5.6645e-12]

# NB: Computations are done in log space so they should be numerically stable.
```

## CLI

<img src="https://github.com/simonepri/lm-scorer/raw/master/media/cli.gif" alt="lm-scorer cli" width="225" align="right"/>

The pip package includes a CLI that you can use to score sentences.

```
usage: lm-scorer [-h] [--model-name MODEL_NAME] [--tokens] [--log-prob]
                 [--reduce REDUCE] [--batch-size BATCH_SIZE]
                 [--significant-figures SIGNIFICANT_FIGURES] [--cuda CUDA]
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
  --reduce REDUCE, -r REDUCE
                        Reduce strategy applied on token probabilities to get
                        the sentence score. Available strategies are: prod,
                        mean, gmean, hmean.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Number of sentences to process in parallel.
  --significant-figures SIGNIFICANT_FIGURES, -sf SIGNIFICANT_FIGURES
                        Number of significant figures to use when printing
                        numbers.
  --cuda CUDA           If provided it runs the model on the given cuda
                        device.
  --debug               If provided it provides additional logging in case of
                        errors.
```


## Development

You can install this library locally for development using the commands below.
If you don't have it already, you need to install [poetry](https://python-poetry.org/docs/#installation) first.

```bash
# Clone the repo
git clone https://github.com/simonepri/lm-scorer
# CD into the created folder
cd lm-scorer
# Create a virtualenv and install the required dependencies using poetry
poetry install
```

You can then run commands inside the virtualenv by using `poetry run COMMAND`.  
Alternatively, you can open a shell inside the virtualenv using `poetry shell`.


If you wish to contribute to this project, run the following commands locally before opening a PR and check that no error is reported (warnings are fine).

```bash
# Run the code formatter
poetry run task format
# Run the linter
poetry run task lint
# Run the static type checker
poetry run task types
# Run the tests
poetry run task test
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

[colab:lm-scorer]: https://colab.research.google.com/github/simonepri/lm-scorer/blob/master/examples/lm_scorer.ipynb

[wiki:language-model]: https://en.wikipedia.org/wiki/Language_model

[github:simonepri]: https://github.com/simonepri
