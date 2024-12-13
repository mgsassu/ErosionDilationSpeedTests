
#include <iostream>
#include <chrono>

#include "ImageFunctions.h"
#include "CudaImageFunctions.cuh"

using namespace std;
using namespace std::chrono;

int main()
{
    // Define new array
    int lowerBound = 256;
    int upperbound = 768;
    int rows = 8192;
    int columns = 8192;
    unsigned char* initialArray = new unsigned char[rows * columns];
    unsigned char* erodedOutput = new unsigned char[rows * columns];
    unsigned char* dilatedOutput = new unsigned char[rows * columns];

    // Initialize input with some values
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {

            if (r > upperbound || r < lowerBound) {
                initialArray[r * columns + c] = 255;
            }

        }
    }

    // Start timing, get execution time for erosion and dilation independently
    auto start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformErosion(rows, columns, initialArray, erodedOutput);
    auto end = chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Erosion time: " << duration.count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformDilation(rows, columns, erodedOutput, dilatedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Dilation time: " << duration.count() << "ms" << endl;



    // Again with the Omp parallelizing
    start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformErosionOmp(rows, columns, initialArray, erodedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Erosion time Omp: " << duration.count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformDilationOmp(rows, columns, erodedOutput, dilatedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Dilation time Omp: " << duration.count() << "ms" << endl;



    // Again, with a thread pool
    start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformErosionThreadPool(rows, columns, initialArray, erodedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Erosion time Thread Pool: " << duration.count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    ImageFunctions::PerformDilationThreadPool(rows, columns, erodedOutput, dilatedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Dilation time Thread Pool: " << duration.count() << "ms" << endl;


    // Finally, with cuda
    start = chrono::high_resolution_clock::now();
    CudaImageFunctions::PerformErosionCUDA(rows, columns, initialArray, erodedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Erosion time CUDA: " << duration.count() << "ms" << endl;

    start = chrono::high_resolution_clock::now();
    CudaImageFunctions::PerformDilationCUDA(rows, columns, erodedOutput, dilatedOutput);
    end = chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Dilation time CUDA: " << duration.count() << "ms" << endl;


    // This reset is in the initial template, so keep it
    cudaDeviceReset();

    // Free all memory
    delete[] initialArray;
    delete[] erodedOutput;
    delete[] dilatedOutput;

    return 0;

}