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
  <a href="https://github.com/simonepri/lm-scorer/actions/workflows/lint.yml">
    <img src="https://github.com/simonepri/lm-scorer/actions/workflows/lint.yml/badge.svg" alt="Lint status" />
  </a>
  <!-- Test - macOS -->
  <a href="https://github.com/simonepri/lm-scorer/actions/workflows/test-macos.yml">
    <img src="https://github.com/simonepri/lm-scorer/actions/workflows/test-macos.yml/badge.svg" alt="Test macOS status" />
  </a>
  <!-- Test - Ubuntu -->
  <a href="https://github.com/simonepri/lm-scorer/actions/workflows/test-ubuntu.yml">
    <img src="https://github.com/simonepri/lm-scorer/actions/workflows/test-ubuntu.yml/badge.svg" alt="Test Ubuntu status" />
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
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Project license" />
  </a>
  <!-- DOI -->
  <a href="https://doi.org/10.5281/zenodo.20584992">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20584992.svg" alt="DOI" />
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

# By default the bos and eos (<|endoftext|>) tokens are added when scoring, so the
# score includes the probability that the sentence ends where it does. Pass
# eos=False (and/or bos=False) to exclude them:
# scorer = LMScorer.from_pretrained("gpt2", device=device, eos=False)

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


## Citation

If you use `lm-scorer` in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff), or the following BibTeX entry:

```bibtex
@software{primarosa_lm_scorer,
  author    = {Primarosa, Simone},
  title     = {{lm-scorer}: Language Model based sentences scoring library},
  year      = {2020},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20584992},
  url       = {https://github.com/simonepri/lm-scorer}
}
```

The concept DOI [`10.5281/zenodo.20584992`](https://doi.org/10.5281/zenodo.20584992)
always resolves to the latest release; each release also has its own version DOI on Zenodo.


## Used in research

`lm-scorer` has been used in the following peer-reviewed publications:

- Schuster, S., & Linzen, T. (2022). *When a sentence does not introduce a discourse entity, Transformer-based models still sometimes refer to it.* NAACL 2022. https://doi.org/10.18653/v1/2022.naacl-main.71
- Gupta, P., Tsvetkov, Y., & Bigham, J. P. (2021). *Synthesizing Adversarial Negative Responses for Robust Response Ranking and Evaluation.* Findings of the ACL: ACL-IJCNLP 2021. https://doi.org/10.18653/v1/2021.findings-acl.338
- Kasner, Z., & Dušek, O. (2020). *Data-to-Text Generation with Iterative Text Editing.* INLG 2020. https://aclanthology.org/2020.inlg-1.9
- Don-Yehiya, S., Choshen, L., & Abend, O. (2022). *PreQuEL: Quality Estimation of Machine Translation Outputs in Advance.* EMNLP 2022. https://doi.org/10.18653/v1/2022.emnlp-main.767
- Alqahtani, A., Sarioglu Kayi, E., Hamidian, S., Compton, M., & Diab, M. (2022). *A Quantitative and Qualitative Analysis of Schizophrenia Language.* CLPsych @ NAACL 2022. https://arxiv.org/abs/2201.10430
- Zhang, Z., Mita, M., & Komachi, M. (2023). *ClozEx: A Task toward Generation of English Cloze Explanation.* Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.347
- Krause, L., Sommerauer, P., & Vossen, P. (2022). *Towards More Informative List Verbalisations.* KGSum @ ISWC 2022. https://ceur-ws.org/Vol-3257/paper14.pdf
- Zhang, J., Mishra, A., Avinesh, P. V. S., Patwardhan, S., & Agarwal, S. (2022). *Can Open Domain Question Answering Systems Answer Visual Knowledge Questions?* arXiv:2202.04306. https://arxiv.org/abs/2202.04306
- Harel, R., Elboher, Y., & Pinter, Y. (2024). *Protecting Privacy in Classifiers by Token Manipulation.* PrivateNLP @ ACL 2024. https://arxiv.org/abs/2407.01334
- Rodrigues, R. C., Inuzuka, M. A., Gomes, J. R. S. A., et al. (2021). *Zero-shot Hashtag Segmentation for Multilingual Sentiment Analysis.* arXiv:2112.03213. https://arxiv.org/abs/2112.03213
- Li, J., Ren, M., Gao, Y., & Yang, Y. (2023). *Ask to Understand: Question Generation for Multi-hop Question Answering.* CCL 2023. https://arxiv.org/abs/2203.09073


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
