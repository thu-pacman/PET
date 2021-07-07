#!/usr/bin/python3
import os
import subprocess

def gemm(b, m, n, k):
    cmd = "./gemm %d %d %d %d" % (b, m, n, k)
    result = subprocess.run(cmd, shell=True, capture_output=True)
    pos = result.stdout.decode().find('best time')
    t = float(result.stdout[pos:].split()[-1])
    return t

def expr_origin():
    t = gemm(1, 512, 768, 768)
    print("origin time: %f" % (t*3))
    return t*3

def expr_opt():
    t = gemm(3, 512, 768, 768)
    print("opt time: %f" % t)
    return t


if __name__ == "__main__":
    t1 = expr_origin()
    t2 = expr_opt()
    print("speedup: %f" % (t1/t2))
