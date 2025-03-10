#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>


using namespace std;

// Function to calculate the mean of the matrix
double calculate_mean(vector<float> &vec)
{
    long double sum = 0.0;
    int totalElements = vec.size();
    for (auto value : vec)
    {
        sum += value;
    }
    return sum / totalElements;
}

// Function to calculate the standard deviation of the matrix
double calculate_standard_deviation(vector<float> &vec, double mean)
{
    long double sum = 0.0;
    long int totalElements = vec.size();
    for (auto value : vec)
    {
        double diff = value - mean;
        sum += diff * diff;
    }
    return sqrt(sum / totalElements);
}

vector<vector<float>> normilize_matrix_z_score(vector<vector<float>> &matrix)
{
    vector<vector<float>> normilized_matrix(matrix.size(), vector<float>(matrix[0].size()));

    for (int row = 0; row < matrix.size(); ++row)
    {
        double mean = calculate_mean(matrix[row]);
        double stddev = calculate_standard_deviation(matrix[row], mean);
        for (int col = 0; col < matrix[0].size(); ++col)
        {
            normilized_matrix[row][col] = (matrix[row][col] - mean) / stddev;
        }
    }
    return normilized_matrix;
}

/** print portion of the matrix first 10 rows, first 10 columns  */
void print_matrix(vector<vector<float>> &matrix)
{
    for (int row = 0; row < 1; ++row)
    {
        for (int col = 0; col < 10; ++col)
        {
            printf("%.2f ", matrix[row][col]);
        }
        printf("\n");
    }
}

int main()
{
    int width = 1000000000;  // Size of the matrix
    int height = 1; // Size of the matrix
    srand(static_cast<unsigned int>(time(0)));
    vector<vector<float>> matrix(height, vector<float>(width));

    // Initialize the matrix with random values
    int end_rand = 10;
    for (int i = 0; i < height; ++i)
    {
        end_rand = round(end_rand * 1.1);
        // We assume at every two there is new datatype ( feature )
        for (int j = 0; j < width; ++j)
        {
            matrix[i][j] = static_cast<float>(rand() % end_rand + 0);
        }
    }

    printf("Matrix before normalization\n");
    print_matrix(matrix);
    printf("Mean of the first row: %.2f\n", calculate_mean(matrix[0]));
    printf("Standard Deviation of the first row: %.2f\n", calculate_standard_deviation(matrix[0], calculate_mean(matrix[0])));
    printf("Mean of the last row: %.2f\n", calculate_mean(matrix[height - 1]));
    printf("Standard Deviation of the last row: %.2f\n", calculate_standard_deviation(matrix[height - 1], calculate_mean(matrix[height - 1])));

    // Mesure the time taken to calculate the standard deviation using AVX2
    clock_t start = clock();
    vector<vector<float>> normilized_matrix = normilize_matrix_z_score(matrix);
    clock_t end = clock();

    double standard_time = double(end - start);
    standard_time = standard_time / CLOCKS_PER_SEC;
    printf("\nMatrix after normalization\n");
    print_matrix(normilized_matrix);
    printf("Mean of the first row: %.2f\n", calculate_mean(normilized_matrix[0]));
    printf("Standard Deviation of the first row: %.2f\n", calculate_standard_deviation(normilized_matrix[0], calculate_mean(normilized_matrix[0])));
    printf("Mean of the last row: %.2f\n", calculate_mean(normilized_matrix[height - 1]));
    printf("Standard Deviation of the last row: %.2f\n", calculate_standard_deviation(normilized_matrix[height - 1], calculate_mean(normilized_matrix[height - 1])));

    printf("Time taken for NON-OpenMP code: %.2f seconds.\n", standard_time);

    return 0;
}


/*

echo -e "-----\n\n\n\n\n ################### ORIGINAL ###################"
g++ -fopenmp ./z_score_norm_original.cpp -o original.exe
for i in {1..20}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./original.exe
done


echo -e "-----\n\n\n\n\n ################### OPENMP ###################"
g++ -fopenmp ./z_score_norm_openMP.cpp -o openMP.exe # && ./openMP.exe
for i in {1..20}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./openMP.exe
done


*/