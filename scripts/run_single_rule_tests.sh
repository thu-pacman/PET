#!/bin/bash
for i in {1..9}; do
    echo "running rule$i"
    ./rule$i @| grep failed
done
