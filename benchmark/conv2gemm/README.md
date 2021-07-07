# conv2gemm

## Algorithm
https://docs.google.com/presentation/d/1UrrEE9oYtumZXhl761MOP840z8kP-zkmanGU4z3S9HM/edit#slide=id.g8d83349b63_0_56

page 4

## gemm.cc
gemm.cc computes a matrix multiplication C = A X B, with A.shape = m x k and B.shape = k x n.
### Compile
```nvcc gemm.cc -lcublas -lcurand -O3 -o gemm```

### run
```./gemm m k n ```

## conv2gemm.cc
to compile the code:
  nvcc conv2gemm.cc -lcurand -lcublas -O3 -o gemm

Please run by typing 
  ./gemm
for the order of the input args

If you want to debug, add the macro DEBUG

example:
  ./gemm 1 3 64 224 224 7 7

TODO:
  To fix some odd bug
  To add a process to remove the redundancy in output

