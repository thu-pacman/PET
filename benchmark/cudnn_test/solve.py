#!/usr/bin/python3

import os

fout = open('data.csv', 'w')
fout.write('group_count,batch_size,input_channel,input_h,input_w,output_channel,kernel_h,kernel_w,form,math_type,\
algo,workspace,memcpy_htod,memcpy_dtoh,convolution,convtotal,total\n')

for file_name in os.listdir('result/'):
    fin = open('result/' + file_name, 'r')
    for line in fin.readlines():
        now = line.strip().split()
        if not now:
            continue
        if now[0] == 'Group':
            group_count, math_type = now[2][:-1], now[5]
        if now[0] == 'Input':
            batch_size, input_channel, input_h, input_w = \
                now[2][:-1], now[3][:-1], now[4][:-1], now[5]
        if now[0] == 'Kernel':
            output_channel, kernel_h, kernel_w = now[2][:-1], now[4][:-1], now[5]
        if now[0] == 'Chosen':
            algo = now[-1]
        if now[0] == 'Workspace':
            workspace = now[-1]
        if now[0] == 'memcpy_htod:':
            memcpy_htod, memcpy_dtoh = now[1], now[4]
        if now[0] == 'choose:':
            convolution, convtotal, total = now[4], now[7], now[10]
    fin.close()
    form = file_name.split('_')[-2][1:]
    fout.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % \
        (group_count, batch_size, input_channel, input_h, input_w, output_channel, kernel_h, kernel_w, form, math_type, \
            algo, workspace, memcpy_htod, memcpy_dtoh, convolution, convtotal, total))
fout.close()
