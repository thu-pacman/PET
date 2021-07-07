import numpy as np

# input with shape (N,C,H,W), kernel with shape (F,C,H,W)
def direct_conv(input, kernel):
    N, C, H, W = input.shape
    F, _, KH, KW = kernel.shape
    OH, OW = H-KH+1, W-KW+1
    # output with shape (N,C,H,W)
    output = np.ndarray(shape=(N,F,OH,OW))
    for n in range(N):
        for f in range(F):
            for h in range(OH):
                for w in range(OW):
                    output[n,f,h,w] = 0
                    for c in range(C):
                        for kh in range(KH):
                            for kw in range(KW):
                                output[n,f,h,w] += input[n,c,h+kh,w+kw] * kernel[f,c,kh,kw]
    return output

# A with shape (M,K), b with shape (K,N)
def matmul(A, B):
    M, K = A.shape
    _, N = B.shape
    output = np.ndarray(shape=(M,N))
    for m in range(M):
        for n in range(N):
            output[m,n] = 0
            for k in range(K):
                output[m,n] += A[m,k] * B[k,n]
    return output

# input with shape (C,N,H,W), kernel with shape (H,W,F,C)
def gemm_conv(input, kernel):
    C, N, H, W = input.shape
    KH, KW, F, _ = kernel.shape
    OH, OW = H-KH+1, W-KW+1
    # output with shape (C,N,H,W)
    output = np.ndarray(shape=(F,N,OH,OW))
    input = input.reshape((C,N*H*W))
    output.fill(0)
    for kh in range(KH):
        for kw in range(KW):
            kernel_tmp = kernel[kh,kw]
            output_tmp = matmul(kernel_tmp, input)
            output_tmp = output_tmp.reshape((F,N,H,W))
            output += output_tmp[:,:,kh:OH+kh,kw:OW+kw]
    return output

# compare two tensors with data layout as nchw and cnhw respectively
def compare(tensor_nchw, tensor_cnhw):
    N, C, H, W = tensor_nchw.shape
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if tensor_nchw[n,c,h,w] != tensor_cnhw[c,n,h,w]:
                        return False
    return True


def run_test(NCHW=(1,3,5,5), FHW=(2,3,3)):
    N, C, H, W = NCHW
    OC, KH, KW = FHW

    input_list = [i for i in range(N*C*H*W)]
    kernel_list = [i for i in range(OC*C*KH*KW)]
    input_nchw = np.ndarray(shape=(N,C,H,W))
    input_cnhw = np.ndarray(shape=(C,N,H,W))
    idx = 0
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    input_nchw[n,c,h,w] = input_list[idx]
                    idx += 1
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    input_cnhw[c,n,h,w] = input_nchw[n,c,h,w]

    kernel_fchw = np.ndarray((OC, C, KH, KW))
    kernel_hwfc = np.ndarray((KH, KW, OC, C))
    idx = 0
    for f in range(OC):
        for c in range(C):
            for h in range(KH):
                for w in range(KW):
                    kernel_fchw[f,c,h,w] = kernel_list[idx]
                    idx += 1
    for f in range(OC):
        for c in range(C):
            for h in range(KH):
                for w in range(KW):
                    kernel_hwfc[h,w,f,c] = kernel_fchw[f,c,h,w]

    OH, OW = H-KH+1, W-KW+1
    output_nchw = np.ndarray(shape=(N,OC,OH,OW))
    output_cnhw = np.ndarray(shape=(OC,N,OH,OW))

    # shape = (N,OC,OH,OW)
    ret_direct = direct_conv(input_nchw, kernel_fchw)
    # shape = (OC, N, OH, OW)
    ret_gemm = gemm_conv(input_cnhw, kernel_hwfc)

    # TODO: compare ret
    same = compare(ret_direct, ret_gemm)
    if same:
        print("same")
    else:
        print("not same")

if __name__ == '__main__':
    run_test((2,4,6,6), (8,3,3))
