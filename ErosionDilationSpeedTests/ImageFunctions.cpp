
#include "ImageFunctions.h"
#include "ThreadPool.h"
#include <omp.h>

using namespace std;

void ImageFunctions::PerformErosion(int rows, int columns, unsigned char* input, unsigned char* output) {
    for (int r = 1; r < rows - 1; r++) {
        for (int c = 1; c < columns - 1; c++) {

            unsigned char minValue = 255;
            // little loop through the surrounding 9 pixels
            for (int r_local = r - 1; r_local < r + 2; r_local++) {
                for (int c_local = c - 1; c_local < c + 2; c_local++) {
                    if (input[r_local * columns + c_local] < minValue) {
                        minValue = input[r_local * columns + c_local];
                    }
                }
            }

            output[r * columns + c] = minValue;
        }
    }
}

void ImageFunctions::PerformDilation(int rows, int columns, unsigned char* input, unsigned char* output) {
    
    
    for (int r = 1; r < rows - 1; r++) {
        for (int c = 1; c < columns - 1; c++) {

            unsigned char maxValue = 0;
            // little loop through the surrounding 9 pixels
            for (int r_local = r - 1; r_local < r + 2; r_local++) {
                for (int c_local = c - 1; c_local < c + 2; c_local++) {
                    if (input[r_local * columns + c_local] > maxValue) {
                        maxValue = input[r_local * columns + c_local];
                    }
                }
            }

            output[r * columns + c] = maxValue;
        }
    }
}

void ImageFunctions::PerformErosionOmp(int rows, int columns, unsigned char* input, unsigned char* output) {
#pragma omp parallel for
    for (int r = 1; r < rows - 1; r++) {
        for (int c = 1; c < columns - 1; c++) {

            unsigned char minValue = 255;
            // little loop through the surrounding 9 pixels
            for (int r_local = r - 1; r_local < r + 2; r_local++) {
                for (int c_local = c - 1; c_local < c + 2; c_local++) {
                    if (input[r_local * columns + c_local] < minValue) {
                        minValue = input[r_local * columns + c_local];
                    }
                }
            }

            output[r * columns + c] = minValue;
        }
    }
}

void ImageFunctions::PerformDilationOmp(int rows, int columns, unsigned char* input, unsigned char* output) {
#pragma omp parallel for
    for (int r = 1; r < rows - 1; r++) {
        for (int c = 1; c < columns - 1; c++) {

            unsigned char maxValue = 0;
            // little loop through the surrounding 9 pixels
            for (int r_local = r - 1; r_local < r + 2; r_local++) {
                for (int c_local = c - 1; c_local < c + 2; c_local++) {
                    if (input[r_local * columns + c_local] > maxValue) {
                        maxValue = input[r_local * columns + c_local];
                    }
                }
            }

            output[r * columns + c] = maxValue;
        }
    }
}

void ImageFunctions::PerformErosionThreadPool(int rows, int columns, unsigned char* input, unsigned char* output) {
    int numThreads = std::thread::hardware_concurrency();  // Use hardware concurrency
    ThreadPool pool(numThreads);

    // Divide the rows among the threads
    auto worker = [&](int startRow, int endRow) {
        for (int r = startRow; r < endRow; r++) {
            for (int c = 1; c < columns - 1; c++) {
                unsigned char minValue = 255;

                // little loop through the surrounding 9 pixels
                for (int r_local = r - 1; r_local < r + 2; r_local++) {
                    for (int c_local = c - 1; c_local < c + 2; c_local++) {
                        if (input[r_local * columns + c_local] < minValue) {
                            minValue = input[r_local * columns + c_local];
                        }
                    }
                }

                output[r * columns + c] = minValue;
            }
        }
        };

    // Enqueue tasks to the thread pool
    int chunkSize = (rows - 2) / numThreads;  // Divide rows into chunks
    for (int i = 0; i < numThreads; ++i) {
        int startRow = 1 + i * chunkSize;
        int endRow = (i == numThreads - 1) ? rows - 1 : startRow + chunkSize;
        pool.enqueueTask([=]() { worker(startRow, endRow); });
    }
}

void ImageFunctions::PerformDilationThreadPool(int rows, int columns, unsigned char* input, unsigned char* output) {
    int numThreads = std::thread::hardware_concurrency();  // Use hardware concurrency
    ThreadPool pool(numThreads);

    // Divide the rows among the threads
    auto worker = [&](int startRow, int endRow) {
        for (int r = startRow; r < endRow; r++) {
            for (int c = 1; c < columns - 1; c++) {
                unsigned char maxValue = 0;

                // little loop through the surrounding 9 pixels
                for (int r_local = r - 1; r_local < r + 2; r_local++) {
                    for (int c_local = c - 1; c_local < c + 2; c_local++) {
                        if (input[r_local * columns + c_local] > maxValue) {
                            maxValue = input[r_local * columns + c_local];
                        }
                    }
                }

                output[r * columns + c] = maxValue;
            }
        }
        };

    // Enqueue tasks to the thread pool
    int chunkSize = (rows - 2) / numThreads;  // Divide rows into chunks
    for (int i = 0; i < numThreads; ++i) {
        int startRow = 1 + i * chunkSize;
        int endRow = (i == numThreads - 1) ? rows - 1 : startRow + chunkSize;
        pool.enqueueTask([=]() { worker(startRow, endRow); });
    }
}