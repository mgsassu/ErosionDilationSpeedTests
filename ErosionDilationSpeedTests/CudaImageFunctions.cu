
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaImageFunctions.cuh"

#include <stdio.h>
#include <algorithm>

// Kernel for erosion
__global__ void erosionKernel(int rows, int columns, const unsigned char* input, unsigned char* output)
{
   // Calculate row and column index for the current thread
   int r = blockIdx.y * blockDim.y + threadIdx.y;
   int c = blockIdx.x * blockDim.x + threadIdx.x;

   // Still ignoring the edges
   if (r > 0 && r < rows - 1 && c > 0 && c < columns - 1) {
      unsigned char minValue = 255;

      // little loop through the surrounding 9 pixels
      for (int r_local = r - 1; r_local <= r + 1; r_local++) {
         for (int c_local = c - 1; c_local <= c + 1; c_local++) {
            if (input[r_local * columns + c_local] < minValue) {
               minValue = input[r_local * columns + c_local];
            }
         }
      }

      output[r * columns + c] = minValue;
   }
}

// Kernel for dilation
__global__ void dilationKernel(int rows, int columns, const unsigned char* input, unsigned char* output)
{
   // Calculate row and column index for the current thread
   int r = blockIdx.y * blockDim.y + threadIdx.y;
   int c = blockIdx.x * blockDim.x + threadIdx.x;

   // Still ignoring the edges
   if (r > 0 && r < rows - 1 && c > 0 && c < columns - 1) {
      unsigned char maxValue = 0;

      // little loop through the surrounding 9 pixels
      for (int r_local = r - 1; r_local <= r + 1; r_local++) {
         for (int c_local = c - 1; c_local <= c + 1; c_local++) {
            if (input[r_local * columns + c_local] > maxValue) {
               maxValue = input[r_local * columns + c_local];
            }
         }
      }

      output[r * columns + c] = maxValue;
   }
}


void CudaImageFunctions::PerformErosionCUDA(int rows, int columns, const unsigned char* input, unsigned char* output)
{
   size_t imageSize = rows * columns * sizeof(unsigned char);

   unsigned char* d_input = nullptr;
   unsigned char* d_output = nullptr;

   // Allocate memory on the GPU
   cudaMalloc((void**)&d_input, imageSize);
   cudaMalloc((void**)&d_output, imageSize);

   // Copy the input data from host to device
   cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

   // Define thread block and grid sizes
   dim3 blockSize(16, 16); // Each block has 16x16 threads
   dim3 gridSize((columns + blockSize.x - 1) / blockSize.x,
      (rows + blockSize.y - 1) / blockSize.y);

   // Launch the kernel
   erosionKernel << <gridSize, blockSize >> > (rows, columns, d_input, d_output);

   // Wait for the kernel to finish
   cudaDeviceSynchronize();

   // Copy the result back to the host
   cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

   // Free GPU memory
   cudaFree(d_input);
   cudaFree(d_output);
}


void CudaImageFunctions::PerformDilationCUDA(int rows, int columns, const unsigned char* input, unsigned char* output)
{
   size_t imageSize = rows * columns * sizeof(unsigned char);

   unsigned char* d_input = nullptr;
   unsigned char* d_output = nullptr;

   // Allocate memory on the GPU
   cudaMalloc((void**)&d_input, imageSize);
   cudaMalloc((void**)&d_output, imageSize);

   // Copy the input data from host to device
   cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

   // Define thread block and grid sizes
   dim3 blockSize(16, 16); // Each block has 16x16 threads
   dim3 gridSize((columns + blockSize.x - 1) / blockSize.x,
      (rows + blockSize.y - 1) / blockSize.y);

   // Launch the kernel
   dilationKernel << <gridSize, blockSize >> > (rows, columns, d_input, d_output);

   // Wait for the kernel to finish
   cudaDeviceSynchronize();

   // Copy the result back to the host
   cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

   // Free GPU memory
   cudaFree(d_input);
   cudaFree(d_output);
}
