all: conv gemm

conv:
	nvcc conv.cu -O3 -lcublas -lcudnn -lcurand -o conv

gemm:
	nvcc gemm.cu -O3 -lcublas -lcudnn -lcurand -o gemm

clean:
	rm -f conv gemm
