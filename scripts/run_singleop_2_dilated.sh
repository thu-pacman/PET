#!/bin/bash
for bs in 1 4 16 64; do
    time -p ./op5 $bs 128 14 64 | tee log.conv5.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op5 $bs 256 14 128 | tee log.conv6.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op5 $bs 512 14 256 | tee log.conv7.$bs;
done
for bs in 1 4 16 64; do
    time -p ./op5 $bs 512 14 512 | tee log.conv8.$bs;
done
