models=(resnet18 csrnet inception bert)
batchsizes=(1 16)
# Generate ONNX 
for i in `seq 0 $((${#models[@]}-1))`; do
    for j in `seq 0 $((${#batchsizes[@]}-1))`; do
        echo Generate ${models[i]} batchsize = ${batchsizes[j]}
        python3 export_as_onnx.py --model ${models[i]} --batchsize ${batchsizes[j]}
    done
done