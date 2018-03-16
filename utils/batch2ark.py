#!/usr/bin/env python3

import sys

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

