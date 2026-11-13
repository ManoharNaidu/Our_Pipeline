"""Shared command-line plumbing for the scripts in this folder."""

from __future__ import annotations

import argparse

from our_pipeline.config import deep_merge, load_config, parse_overrides


def parser(description: str) -> argparse.ArgumentParser:
    made = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    made.add_argument("--config", required=True, help="a file in configs/datasets/")
    made.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. training.max_epochs=2")
    return made


def config_from(args) -> dict:
    return deep_merge(load_config(args.config), parse_overrides(args.set))
