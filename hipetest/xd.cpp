__global__ void myKernel(int N, double *d_a){
    int i = threadId.x + blockId.x * blockDim.x;
    if (i < N) {
        d_a[i] *= 2.0;
    }
}

int main() {
    dim3 threads(256, 1, 1);
    dim3 blocks((N + 256 - 1)/256, 1, 1);

    
}