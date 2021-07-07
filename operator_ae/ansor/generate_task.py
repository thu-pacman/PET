import re
with open('task.txt') as f:
    for l in f.readlines():
        pattern = re.compile(r'\[.*?\]')   # 查找数字
        result = pattern.findall(l)
        print([eval(str(result[i])) for i in range(len(result))], end=',\n')
# N, H, W, CO, CI, KH, KW, strides, padding, dilation = 1, 14, 14, 512, 512, 3, 3, (
#     1, 1), (3, 3), 3
