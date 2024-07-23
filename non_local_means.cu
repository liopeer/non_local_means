#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__global__ void non_local_means(
    float* img, 
    float* out, 
    int height, 
    int width, 
    int radius
) {
    // coordinates of the center pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("x: %d, y: %d\n", x, y);

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int x1 = x + i;
            int y1 = y + j;
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                float d = distance(x, y, x1, y1);
                out[x * height + y] += img[x1 * height + y1] * exp(-d);
            }
        }
    }
}

extern "C" {
    void interface_nlm(float* img, float* out, int height, int width, int radius) {
        float* d_img;
        float* d_out;

        cudaMalloc(&d_img, sizeof(float) * height * width);
        cudaMalloc(&d_out, sizeof(float) * height * width);

        cudaMemcpy(d_img, img, sizeof(float) * height * width, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, out, sizeof(float) * height * width, cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(width / blockDim.y), ceil(height / blockDim.x));

        non_local_means<<<gridDim, blockDim>>>(d_img, d_out, height, width, radius);
        cudaDeviceSynchronize();

        cudaMemcpy(out, d_out, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

        cudaFree(d_img);
        cudaFree(d_out);
    }
}