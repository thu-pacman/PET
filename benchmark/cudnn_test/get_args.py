#!/usr/bin/python3

import os

fout = open('all_args.sum', 'w')

def Get_Args(filename):
    fin = open(filename, 'r')
    st = 0
    for line in fin.readlines():
        now = line.strip().split()
        if now and st == 1 and now[0][0:6] == 'Conv2d':
            fout.write('%s_%s_%s\n' % (now[1], now[2], now[3]))
        if now and now[0][0] == '=':
            st += 1
        if st == 2:
            break
    fin.close()

def main():
    filedir = 'summary/'
    for filename in os.listdir(filedir):
        if filename[-4:] == '.sum':
            Get_Args(filedir + filename)
    fout.close()
    
if __name__ == '__main__':
    main()

