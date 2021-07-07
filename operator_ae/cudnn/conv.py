#!/usr/bin/python3
import os
import subprocess

def conv(n, c, h, w, f, wc, r, s, ph, pw, sh, sw, dh, dw):
    cmd = "./conv -n %d -c %d -h %d -w %d -f %d -g %d -r %d -s %d -ph %d -pw %d -sh %d -sw %d -dh %d -pw %d" % (n, c, h, w, f, c/wc, r, s, ph, pw, sh, sw, dh, dw)
    result = subprocess.run(cmd, shell=True, capture_output=True)
    pos = result.stdout.decode().find('best time')
    t = float(result.stdout[pos:].split()[-1])
    return t

def expr_origin():
    t = conv(1, 48, 38, 38, 64, 48, 5, 5, 2, 2, 1, 1, 1, 1)
    print("origin time: %f" % t)
    return t

def expr_opt():
    t = conv(16, 48, 10, 10, 64, 48, 5, 5, 2, 2, 1, 1, 1, 1)
    print("opt time: %f" % t)
    return t

if __name__ == "__main__":
    t1 = expr_origin()
    t2 = expr_opt()
    print("speedup: %f" % (t1/t2))
