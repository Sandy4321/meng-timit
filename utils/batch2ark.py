#!/usr/bin/env python3

import sys

# Override default pipe handling so that Python doesn't start writing to closed pipe
# and cause a BrokenPipeError
# See https://github.com/python/mypy/issues/2893 for explanation
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

area = 'head'

for line in sys.stdin:
    if area == 'head':
        area = 'body'
        print('{}  ['.format(line.strip()), end='')

    elif area == 'body' and line == '.\n':
        area = 'head'
        print(' ]')

    elif area == 'body':
        print('')
        print('  ', end='')
        print(line.strip(), end='')

