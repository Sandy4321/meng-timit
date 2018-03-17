#!/usr/bin/env python3

import sys

# Override default pipe handling so that Python doesn't start writing to closed pipe
# and cause a BrokenPipeError
# See https://github.com/python/mypy/issues/2893 for explanation
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

for line in sys.stdin:
    parts = line.split()

    print(parts[0])
    print(' '.join(parts[1:]))
    print('.')
