import argparse
import dataclasses
import json
import sys
from pathlib import Path

from transformers import HfArgumentParser


class Config:
    def dump(self, file: str | Path) -> None:
        d = dataclasses.asdict(self)
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w') as fp:
            json.dump(d, fp, indent=2)

    @classmethod
    def load(cls):
        if '--config' in sys.argv:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', required=True)
            args = parser.parse_args()
            return cls.load_from_json(file=args.config)

        return cls.load_from_cli()

    @classmethod
    def load_from_cli(cls):
        parser = HfArgumentParser(cls)
        config, unknown = parser.parse_args_into_dataclasses(
            return_remaining_strings=True,
        )
        if unknown:
            raise ValueError(f'Unknown command line arguments: {unknown}')

        return config

    @classmethod
    def load_from_json(cls, file: str):
        with open(file) as fp:
            data = json.load(fp)

        data = [e for k, v in data.items() for e in [f'--{k}', str(v)]]

        parser = HfArgumentParser(cls)
        config, unknown = parser.parse_args_into_dataclasses(
            args=data,
            return_remaining_strings=True,
        )
        if unknown:
            raise ValueError(f'Unknown json arguments: {unknown}')

        return config
