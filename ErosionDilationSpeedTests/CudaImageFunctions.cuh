#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

class CudaImageFunctions {
public:

   static void PerformErosionCUDA(int rows, int columns, const unsigned char* input, unsigned char* output);

   static void PerformDilationCUDA(int rows, int columns, const unsigned char* input, unsigned char* output);
};