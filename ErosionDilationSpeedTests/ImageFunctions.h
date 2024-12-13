#pragma once

using namespace std;

class ImageFunctions
{
public:
    static void PerformErosion(int rows, int columns, unsigned char* input, unsigned char* output);

    static void PerformDilation(int rows, int columns, unsigned char* input, unsigned char* output);

    static void PerformErosionOmp(int rows, int columns, unsigned char* input, unsigned char* output);

    static void PerformDilationOmp(int rows, int columns, unsigned char* input, unsigned char* output);

    static void PerformErosionThreadPool(int rows, int columns, unsigned char* input, unsigned char* output);

    static void PerformDilationThreadPool(int rows, int columns, unsigned char* input, unsigned char* output);
};

