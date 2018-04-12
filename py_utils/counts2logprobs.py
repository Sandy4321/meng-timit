#!/usr/bin/env python3

import math
import sys

# Override default pipe handling so that Python doesn't start writing to closed pipe
# and cause a BrokenPipeError
# See https://github.com/python/mypy/issues/2893 for explanation
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

for line in sys.stdin:
    cleaned_line = line.replace("[", "").replace("]", "")
    counts = list(map(float, filter(lambda x: len(x) > 0 and not x.isspace(), cleaned_line.split())))
    total_count = sum(counts)
    log_probs = list(map(lambda x: math.log(x / total_count), counts))
    print(" ".join(map(str, log_probs)))
