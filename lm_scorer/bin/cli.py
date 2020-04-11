#!/usr/bin/env python3

from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import argparse
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


def main(args: argparse.Namespace) -> None:
    if args.sentences_file_path == "-":
        sentences_stream = sys.stdin
    else:
        sentences_stream = open(args.sentences_file_path, "r")

    device = torch.device("cuda:%d" % args.cuda if args.cuda >= 0 else "cpu")
    scorer = LMScorer.from_pretrained(args.model_name, device=device)

    if args.tokens:
        for sentence in sentences_stream:
            sentence = sentence.strip()
            _, token_scores = scorer.score(
                sentence, return_log_prob=args.log_prob, return_tokens=True
            )  # type: Tuple[float, Dict[str, float]]  # type: ignore
            for token, score in token_scores.items():
                print("%s\t%.5g" % (token, score))
            print("")
    else:
        for sentence in sentences_stream:
            sentence = sentence.strip()
            sent_score = scorer.score(
                sentence, return_log_prob=args.log_prob
            )  # type: float  # type: ignore
            print("%s\t%.5g" % (sentence, sent_score))

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
