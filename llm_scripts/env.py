import os


def is_ddp() -> bool:
    return 'LOCAL_RANK' in os.environ


def is_primary() -> bool:
    return rank() == 0


def rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))
