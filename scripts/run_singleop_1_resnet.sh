#!/bin/bash
for bs in 1 4 16 64; do
    time -p ./op4 $bs 64 56 64 | tee log.conv1.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op4 $bs 128 28 128 | tee log.conv2.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op4 $bs 256 14 256 | tee log.conv3.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op4 $bs 512 7 512 | tee log.conv4.$bs;
done
