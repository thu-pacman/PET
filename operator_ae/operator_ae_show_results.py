import os
import subprocess

if __name__ == "__main__":
    print("OPERATOR on TVM")
    os.chdir("/home/osdi_ae/osdi21-tvm");
    cmd = "./show_results.sh"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    data = result.stdout.decode()
    print(data)
    data_tvm = {}
    result_tvm = {}
    op = ""
    for line in data.split("\n"):
        if line[:8] == "========":
            if line[16:] == "Dilated Conv":
                op = "dilated_conv"
            if line[16:] == "Group Conv":
                op = "group_conv"
            if line[16:] == "Conv":
                op = "conv"
            if line[16:] == "Batch GEMM":
                op = "matmul"
        if (line[:8] == "BestTime"):
            if op not in data_tvm:
                data_tvm[op] = []
            time = line.split("=")[1].strip()
            if time[-2:] == "ms":
                time = time[:-2]
            time = float(time)
            data_tvm[op].append(time)

    op = "dilated_conv"
    time0 = data_tvm[op][0]
    time1 = min(data_tvm[op])
    result_tvm[op] = [time0, time1]
    op = "group_conv"
    time0 = data_tvm[op][0] * 2 + data_tvm[op][1] * 2
    time1 = data_tvm[op][2] + data_tvm[op][3]
    result_tvm[op] = [time0, time1]
    op = "conv"
    time0 = data_tvm[op][0]
    time1 = min(data_tvm[op])
    result_tvm[op] = [time0, time1]
    op = "matmul"
    time0 = data_tvm[op][0] * 3
    time1 = data_tvm[op][1]
    result_tvm[op] = [time0, time1]
    
    print("OPERATOR on Ansor")
    os.chdir("/home/osdi_ae/osdi21-ansor");
    cmd = "./show_results.sh"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    data = result.stdout.decode()
    print(data)
    data_ansor = {}
    result_ansor = {}
    filelist = ["conv-2-nico3.txt", "gconv-1-nico1.txt", "conv-5-nico1.txt", "conv-5-nico2.txt", "bmm-1-nico1.txt", "task.txt"]
    op = ""
    filename = ""
    for line in data.split("\n"):
        if line.strip()[-4:] == ".txt":
            filename = line.strip()
            data_ansor[filename] = []
            continue
        if filename == "task.txt":
            if len(line) > 0 and len(line.split(" ")) == 1:
                data_ansor[filename].append(float(line.strip()))
            continue
        if len(line) > 0 and line[0] == "|":
            tmp = line.strip().split("|")
            if tmp[1].strip() != "ID":
                time = tmp[2].strip()
                data_ansor[filename].append(float(time))
    
    op = "dilated_conv"
    op_time = []
    op_file = ["conv-5-nico1.txt", "conv-5-nico1.txt", "conv-5-nico1.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt", "conv-5-nico2.txt"]
    op_id = [7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(12):
        op_time.append(data_ansor[op_file[i]][op_id[i]])
    time0 = op_time[0]
    time1 = min(op_time)
    result_ansor[op] = [time0, time1]
    
    op = "group_conv"
    op_time = []
    op_file = ["conv-2-nico3.txt", "conv-2-nico3.txt", "gconv-1-nico1.txt", "gconv-1-nico1.txt"]
    op_id = [1, 2, 0, 1]
    for i in range(4):
        op_time.append(data_ansor[op_file[i]][op_id[i]])
    time0 = op_time[0] * 2 + op_time[1] * 2
    time1 = min([op_time[0] * 2, op_time[2]]) + min([op_time[1] * 2, op_time[3]])
    result_ansor[op] = [time0, time1]

    op = "conv"
    op_time = []
    op_file = ["conv-5-nico1.txt" for _ in range(7)]
    op_id = range(7)
    for i in range(7):
        op_time.append(data_ansor[op_file[i]][op_id[i]])
    time0 = op_time[0];
    time1 = min(op_time)
    result_ansor[op] = [time0, time1]

    op = "matmul"
    time0 = data_ansor["bmm-1-nico1.txt"][0] * 3
    time1 = min(data_ansor["task.txt"]) * 1000
    result_ansor[op] = [time0, time1]
    os.chdir("/home/osdi_ae/pet-osdi21-ae/operator_ae/cudnn")
    # cmd = "cat result.txt"
    cmd = "./run.sh"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    data = result.stdout.decode()
    cudnn_time = {}
    result_cudnn = {}

    op = ""
    for line in data.split("\n"):
        if line.strip() == "dilated conv":
            op = "dilated_conv"
        if line.strip() == "group conv":
            op = "group_conv"
        if line.strip() == "conv":
            op = "conv"
        if line.strip() == "gemm":
            op = "matmul"
        if op != "" and op not in result_cudnn:
            result_cudnn[op] = []
        tmp = line.strip().split()
        if len(tmp) > 2:
            if tmp[0] == "origin":
                result_cudnn[op].append(float(tmp[-1]))
            if tmp[0] == "opt":
                result_cudnn[op].append(float(tmp[-1]))

    op_list = ["dilated_conv", "group_conv", "conv", "matmul"]
    print("Time: ")
    for op in op_list:
        print("%-12s:\tbefore\tafter"%op)
        print("cudnn:\t\t%.3f\t%.3f"%(result_cudnn[op][0], result_cudnn[op][1]))
        print("tvm:\t\t%.3f\t%.3f"%(result_tvm[op][0], result_tvm[op][1]))
        print("ansor:\t\t%.3f\t%.3f"%(result_ansor[op][0], result_ansor[op][1]))

    print("Speedup: ")
    for op in op_list:
        print("%-12s:\tbefore\tafter"%op)
        print("cudnn:\t\t%.3f\t%.3f"%(result_cudnn[op][0] / result_cudnn[op][0], result_cudnn[op][0] / result_cudnn[op][1]))
        print("tvm:\t\t%.3f\t%.3f"%(result_cudnn[op][0] / result_tvm[op][0], result_cudnn[op][0] / result_tvm[op][1]))
        print("ansor:\t\t%.3f\t%.3f"%(result_cudnn[op][0] / result_ansor[op][0], result_cudnn[op][0] / result_ansor[op][1]))
