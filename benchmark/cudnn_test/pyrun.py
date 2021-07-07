#!/usr/bin/python3

import os

fin = open('all_args.sum', 'r')
for line in fin.readlines():
    now = line.strip().split('_')
    if not now:
        break
    input_size = now[0][1:-1].split(',')
    input_channel = input_size[1]
    input_height  = input_size[2]
    input_weight  = input_size[3]

    kernel_size = now[-1][1:-1].split(',')
    kernel_height  = kernel_size[2]
    kernel_weight  = kernel_size[3]
    output_channel = kernel_size[1]

    kernel_pad = int(kernel_height) // 2

    for batch_size in [int(4**k) for k in range(6)]:
        run_precmd = './conv -gc 1 -bs %d -c %s -kernels %s -inh %s -inw %s -kerh %s -kerw %s -padh %d -padw %d' % \
            (batch_size, input_channel, output_channel, input_height, input_weight, kernel_height, kernel_weight, kernel_pad, kernel_pad)
        for form in [' -nchw']: #[' -nhwc', ' -nchw']:
            for math_type in [' -default-math', ' -tensor-op-math', ' -tensor-op-math-allow-conversion']:
               run_cmd = run_precmd + form + math_type
               file_name = run_cmd[2:].replace(' ', '_')
               run_cmd = run_cmd + ' > result/%s.dat' % file_name
               print(run_cmd)
               os.system(run_cmd)

fin.close()
