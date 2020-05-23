#!/usr/bin/env python3

from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import argparse
import itertools
import os
import sys

import torch

from ..models.auto import AutoLMScorer as LMScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get sentences probability using a language model.",
    )
    parser.add_argument(
        "sentences_file_path",
        metavar="sentences-file-path",
        type=str,
        help="A file containing sentences to score, one per line."
        " If - is given as filename it reads from stdin instead.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="gpt2",
        help="The pretrained language model to use. Can be one of: %s."
        % ", ".join(LMScorer.supported_model_names()),
    )
    parser.add_argument(
        "--tokens",
        "-t",
        action="store_true",
        help="If provided it provides the probability of each token of each sentence.",
    )
    parser.add_argument(
        "--log-prob",
        "-lp",
        action="store_true",
        help="If provided log probabilities are returned instead.",
    )
    parser.add_argument(
        "--reduce",
        "-r",
        type=str,
        default="prod",
        help="Reduce strategy applied on token probabilities to get the sentence score."
        " Available strategies are: prod, mean, gmean, hmean.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Number of sentences to process in parallel.",
    )
    parser.add_argument(
        "--significant-figures",
        "-sf",
        type=int,
        default=5,
        help="Number of significant figures to use when printing numbers.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="If provided it runs the model on the given cuda device.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If provided it provides additional logging in case of errors.",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    if args.sentences_file_path != "-":
        args.sentences_file_path = os.path.realpath(args.sentences_file_path)


def validate_args(args: argparse.Namespace) -> None:
    if args.sentences_file_path != "-":
        if not os.path.isfile(args.sentences_file_path):
            raise ValueError("The provided sentences file path is invalid.")

    if args.cuda >= 0 and not torch.cuda.is_available():
        raise ValueError("No Cuda device found.")

    if args.cuda >= torch.cuda.device_count():
        device_count = torch.cuda.device_count()
        raise ValueError("Invalid Cuda device: %d/%d." % (args.cuda, device_count))

    if args.batch_size <= 0:
        raise ValueError("The batch size must be positive.")

    if args.significant_figures <= 0:
        raise ValueError("The number of significant figures must be positive.")


T1 = TypeVar("T1")  # pylint: disable=invalid-name


def grouper(iterable: Iterable[T1], size: int) -> Generator[List[T1], None, None]:
    it = iter(iterable)  # pylint: disable=invalid-name
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def main(args: argparse.Namespace) -> None:
    # pylint: disable=too-many-locals
    if args.sentences_file_path == "-":
        sentences_stream = sys.stdin
    else:
        sentences_stream = open(args.sentences_file_path, "r")

    sig_fig = args.significant_figures
    batch_size = args.batch_size
    device = torch.device("cuda:%d" % args.cuda if args.cuda >= 0 else "cpu")
    scorer = LMScorer.from_pretrained(
        args.model_name, device=device, batch_size=batch_size
    )

    buffer_size = args.batch_size * 2
    for sentences in grouper(sentences_stream, buffer_size):
        sentences = [sentence.strip() for sentence in sentences]

        sent_scores = scorer.sentence_score(
            sentences, log=args.log_prob, reduce=args.reduce
        )
        if args.tokens:
            sent_info = scorer.tokens_score(sentences, log=args.log_prob)

        sent_num = len(sentences)
        for i in range(sent_num):
            sentence, sent_score = sentences[i], sent_scores[i]
            print(f"%s\t%.{sig_fig}g" % (sentence, sent_score))
            if args.tokens:
                scores, _, tokens = sent_info[i]
                for score, token in zip(scores, tokens):
                    print(f"%s\t%.{sig_fig}g" % (token, score))
                print("")

    if args.sentences_file_path != "-":
        sentences_stream.close()


def run() -> None:
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except KeyboardInterrupt:
        print("\nAborted!")
    except Exception as err:  # pylint: disable=broad-except
        if args.debug:
            raise
        print("Error: %s" % err)


if __name__ == "__main__":
    run()
