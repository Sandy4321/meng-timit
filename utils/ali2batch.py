#!/usr/bin/env python3

import sys

for line in sys.stdin:
    parts = line.split()

    print(parts[0])
    print(' '.join(parts[1:]))
    print('.')
