"""Main method for interacting with denoiser through CLI.
"""

import sys
import joint
import joint.cli

from typing import List


def start_cli(args: List[str] = None):
    joint.logging_helper.setup()
    if args is not None:
        sys.argv[1:] = args
    joint.cli.start()


if __name__ == "__main__":
    start_cli()
