#include <stdio.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath> 

#pragma GCC target("avx2")

using namespace std;

double calculate_meanAV(vector<float> &vector){
    long double totalSum = 0.0;
    long int totalElements = vector.size(); 
    int width = vector.size();
    int j = 0;
    __m256 sum_8_floats = _mm256_setzero_ps();  // AVX register to hold the sum

    // Using SIMD, 8 floats at a time 
    for (; j <= totalElements - 8; j += 8){
        __m256 values = _mm256_loadu_ps(&vector[j]);  // Load 8 float values
        sum_8_floats = _mm256_add_ps(sum_8_floats, values);    // Sum the float values
    }

    // Convert the value of the 8 floats to one value ( totalSum)
    float sumArray[8];
    _mm256_storeu_ps(sumArray, sum_8_floats);  // Store the 8 values from the AVX register into an array
    for (int i = 0; i < 8; ++i){
        totalSum += sumArray[i];  // Sum the 8 values together
    }

    // Remaining elements (those not divisible by 8)
    for (; j < totalElements; ++j){
        totalSum += vector[j];
    }

    return totalSum / totalElements;
}

double calculate_standard_deviation_av(vector<float> &vector, double mean){
    __m256 sum_8_floats = _mm256_setzero_ps();  // AVX register to hold 8 sum_dff_sqrs
    long double sum_dff_sqrs = 0.0;
    long int totalElements = vector.size(); 
    int j = 0;
    // Using SIMD, 8 floats at a time 
    for (; j <= totalElements - 8; j += 8){
        __m256 values = _mm256_loadu_ps(&vector[j]);  // Load 8 float values
        __m256 difference = _mm256_sub_ps(values, _mm256_set1_ps(mean)); // Subtract 
        __m256 diffSquared = _mm256_mul_ps(difference, difference); // Square
        sum_8_floats = _mm256_add_ps(sum_8_floats, diffSquared);
    }
    
    float sumArray[8]; // Gets the values from sum_8_floats
    _mm256_storeu_ps(sumArray, sum_8_floats); 
    for (int i = 0; i < 8; ++i){
        sum_dff_sqrs += sumArray[i];  // Sum the 8 values together
    }
    // Remaining elements ( not divisible by 8)
    for (; j < totalElements; ++j) {
        sum_dff_sqrs += (vector[j] - mean) * (vector[j] - mean);
    }

    return sqrt(sum_dff_sqrs / totalElements);
}

vector<vector<float>> normilize_matrix_z_score_avx(vector<vector<float>> &matrix){
    vector<vector<float>> normilized_matrix(matrix.size(), vector<float>(matrix[0].size()));
    
    for(int row = 0; row < matrix.size(); ++row){
        int col = 0;
        int totalCols = matrix[0].size();
        double mean = calculate_meanAV(matrix[row]);
        double stnddev = calculate_standard_deviation_av(matrix[row], mean);
        __m256 mean_vect = _mm256_set1_ps(float(mean)); // Broadcast the mean value in 8 floats 
        __m256 stddev_vect = _mm256_set1_ps(float(stnddev)); 

        for(; col <= totalCols - 8; col += 8){
            __m256 data = _mm256_loadu_ps(&matrix[row][col]); // Load 8 floats
            __m256 difference = _mm256_sub_ps(data, mean_vect); // Subtract mean
            __m256 normalized = _mm256_div_ps(difference, stddev_vect); // Divide by stnddev
            _mm256_storeu_ps(&normilized_matrix[row][col], normalized); // STore to matrix 
        }

        //  any remaining 
        for(; col < totalCols; ++col){
            normilized_matrix[row][col] = (matrix[row][col] - mean) / stnddev;
        }
    }
    return normilized_matrix;
}

/** print portion of the matrix first 10 rows, first 10 columns  */
void print_matrix(vector<vector<float>> &matrix){
    for(int row = 0; row < 10; ++row){
        for(int col = 0; col < 10; ++col){
            printf("%.2f ", matrix[row][col]);
        }
        printf("\n");
    }
}

int main(){
    int width = 70 * 512;  // Size of the matrix
    int height = 50 * 512;  // Size of the matrix
    srand(static_cast<unsigned int>(time(0)));
    vector<vector<float>> matrix(height, vector<float>(width));

    // Initialize the matrix with random values, The values ranges change between rows 
    int end_rand = 10; 
    for (int i = 0; i < height; ++i) {
        end_rand = round(end_rand * 1.1);
        // We assume at every two there is new datatype ( feature )  
        for (int j = 0; j < width; ++j){
            matrix[i][j] = static_cast<float>(rand() % end_rand + 0);
        }
    }

    printf("Matrix before normalization\n");
    print_matrix(matrix);
    printf("Mean of the first row: %.2f\n", calculate_meanAV(matrix[0]));
    printf("Standard Deviation of the first row: %.2f\n", calculate_standard_deviation_av(matrix[0], calculate_meanAV(matrix[0])));
    printf("Mean of the last row: %.2f\n", calculate_meanAV(matrix[height - 1]));
    printf("Standard Deviation of the last row: %.2f\n", calculate_standard_deviation_av(matrix[height - 1], calculate_meanAV(matrix[height - 1])));
    
    // Measuring time it takes to normalize the matrix
    clock_t start = clock();
    vector<vector<float>> normilized_matrix = normilize_matrix_z_score_avx(matrix);    
    clock_t end = clock();

    double simd_time = double(end - start);
    simd_time = simd_time / CLOCKS_PER_SEC;
    printf("\nMatrix after normalization\n");
    print_matrix(normilized_matrix);
    printf("Mean of the first row: %.2f\n", calculate_meanAV(normilized_matrix[0]));
    printf("Standard Deviation of the first row: %.2f\n", calculate_standard_deviation_av(normilized_matrix[0], calculate_meanAV(normilized_matrix[0])));
    printf("Mean of the last row: %.2f\n", calculate_meanAV(normilized_matrix[height - 1]));
    printf("Standard Deviation of the last row: %.2f\n", calculate_standard_deviation_av(normilized_matrix[height - 1], calculate_meanAV(normilized_matrix[height - 1])));
    printf("Time taken for SIMD vectorized code: %.2f seconds.\n", simd_time);

    return 0;
}
