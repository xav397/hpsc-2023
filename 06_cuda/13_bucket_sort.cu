#include <cstdio>
#include <cstdlib>
#include <vector>

__global__
void init(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i >= range ) return;
  bucket[i] = 0;
}

__global__
void fill(int *bucket, int *key, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i >= n ) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__
void scan(int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i >= range ) return;
  for(int j=1; j<range; j<<=1) {
    offset[i] = bucket[i];
    __syncthreads();
    bucket[i] += offset[i-j];
    __syncthreads();
  }
}

__global__
void sort(int *bucket, int *key, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=range-1; j>=0; j--){
    if( i >= bucket[j] ) return;
    key[i] = j;
  }
}

int main() {
  int n = 50;
  int range = 5;
  const int M = 1024;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket, *offset, *d_key;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMalloc(&d_key, n*sizeof(int));
  cudaMemcpy(d_key, key.data(), n*sizeof(int), cudaMemcpyHostToDevice);

  init<<<(range+M-1)/M,M>>>(bucket, range);
  cudaDeviceSynchronize();
  //std::vector<int> bucket(range); 
  //for (int i=0; i<range; i++) {
  //  bucket[i] = 0;
  //}

  fill<<<(n+M-1)/M,M>>>(bucket, d_key, n);
  cudaDeviceSynchronize();
  //for (int i=0; i<n; i++) {
  //  bucket[key[i]]++;
  //}

  scan<<<(range+M-1)/M,M>>>(bucket, offset, range);
  cudaDeviceSynchronize();
  sort<<<(range+M-1)/M,M>>>(bucket, d_key, range);
  cudaDeviceSynchronize();
  //for (int i=0, j=0; i<range; i++) {
  //  for (; bucket[i]>0; bucket[i]--) {
  //    key[j++] = i;
  //  }
  //}
  
  cudaMemcpy(key.data(), d_key, n*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
