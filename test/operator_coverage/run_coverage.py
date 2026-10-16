#!/usr/bin/env python3
"""Run the onnx2c operator coverage campaign.

Example::

    python3 test/operator_coverage/run_coverage.py --quick
    python3 test/operator_coverage/run_coverage.py --mode exhaustive
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
