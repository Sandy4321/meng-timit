#!/bin/bash

# This script is used so that we can run lattice-to-ctm-conf in parallel with different
# LM weights. In the old way, the bc command to compute the acoustic scale was evaluated
# before run.pl substituted the actual LM weight into the string "LMWT", so this is a
# workaround to delay evaluation of the bc command.

if [ $# -ne 3 ]; then
    echo "Usage: lattice-to-ctm-conf-parallel.sh <LM weight> <LM scale> <aligned phones ark file>"
    exit 1
fi

lmwt=$1
mbr_scale=$2
aligned_phones=$3

lattice-to-ctm-conf --acoustic-scale=$(bc <<<"scale=8; 1/$lmwt*$mbr_scale") --lm-scale=$mbr_scale ark:$aligned_phones -
