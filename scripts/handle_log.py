def handle(filename):
    ret = []
    with open(filename) as f:
        lines = f.readlines()
        for i in range(len(lines)//2):
            ret.append((float(lines[2*i+1]), lines[2*i+2]))
    best = min(ret, key = lambda item: item[0])
    base = ret[0]
    print("base: %.3f, %s" % base)
    print("best: %.3f, %s" % best)
    print("speedup: %.2fx" % (base[0]/best[0]))

import sys

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python handle_log.py filename")
    else:
        filename = sys.argv[1]
        handle(filename)
        