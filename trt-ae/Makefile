APP := runtrt-4d runtrt-3d

CC := g++

CUDA_INSTALL_PATH ?= /usr/local/cuda-10.2
CUDNN_INSTALL_PATH ?= /usr/local/cuda-10.2/extras/cudnn-7.6.5

TRT_INSTALL_PATH ?= /home/huangkz/TensorRT-7.0.0.11

SRCS := \
	runtrt-4d.cpp runtrt-3d.cpp

OBJS := $(SRCS:.cpp=.o)

CPPFLAGS := \
	-std=c++11 -g\
	-I"$(TRT_INSTALL_PATH)/include" \
	-I"$(CUDA_INSTALL_PATH)/include" \
	-I"$(CUDNN_INSTALL_PATH)/include"

LDFLAGS := \
    -lnvonnxparser \
	-lnvparsers \
	-lnvinfer \
	-lcudart \
	-lcudnn \
	-L"$(TRT_INSTALL_PATH)/lib" \
	-L"$(CUDA_INSTALL_PATH)/lib64" \
	-L"$(CUDNN_INSTALL_PATH)/lib64"

all: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	@ $(CC) $(CPPFLAGS) -c $<

runtrt-3d: runtrt-3d.o
	@echo "Linking: $@"
	@ $(CC) -o $@ $< $(CPPFLAGS) $(LDFLAGS)

runtrt-4d: runtrt-4d.o
	@echo "Linking: $@"
	@ $(CC) -o $@ $< $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS)
