#!/bin/bash

# Bash Script which creates the directory of the most recent backup data and send the
# Data from the simulations there

max=$(ls -d  BackupData*/ 2>/dev/null | sed 's|BackupData||; s|/||' | sort -n | tail -1)

max=${max:-0}

mkdir "BackupData$((max+1))"

ls *.txt | xargs -I{} mv {} BackupData$((max+1))/