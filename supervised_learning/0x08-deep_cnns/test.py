#!/usr/bin/env python3
con = 0
linesB = {hash(line) for line in open('intranet.txt')}
for line in open('output.txt'):
    if hash(line) not in linesB:
        print('line: {}'.format(con))
        print('diferente')
    con += 1
