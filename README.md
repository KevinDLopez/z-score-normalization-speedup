# Z-Score Normalization Speedup

This project explores various optimization techniques to accelerate z-score normalization, a common preprocessing step in data analysis and machine learning. The implementations compare standard sequential processing against optimized versions using parallel processing (OpenMP) and vectorization (SIMD).

## What is Z-Score Normalization?

Z-score normalization (also known as standardization) transforms data to have a mean of 0 and a standard deviation of 1. For each value in a dataset, it applies the formula:

```
z = (x - μ) / σ
```

Where:
- `x` is the original value
- `μ` is the mean of the dataset
- `σ` is the standard deviation of the dataset

This transformation is computationally intensive for large datasets, making it a good candidate for optimization.

## Project Structure

```
├── LICENSE
├── OpenMP/                       # OpenMP-based parallel implementation
│   ├── averages.ipynb            # Analysis of performance results
│   ├── openMP.log                # Performance logs for OpenMP version
│   ├── original.log              # Performance logs for baseline version
│   ├── report.pdf                # Detailed report on OpenMP optimization
│   ├── report.tex                # LaTeX source for the report
│   └── src/
│       ├── z_score_norm_openMP.cpp    # OpenMP implementation
│       └── z_score_norm_original.cpp  # Original baseline implementation
└── SIMD/                         # SIMD-based vectorized implementation
    ├── averages.ipynb            # Analysis of performance results
    ├── output_SIMD.txt           # Performance output for SIMD version
    ├── output_orignal.txt        # Performance output for baseline version
    ├── report.pdf                # Detailed report on SIMD optimization
    ├── report.tex                # LaTeX source for the report
    ├── Screenshots/              # Visual performance comparisons
    └── src/
        ├── z_score_norm__Optimized__.cpp  # SIMD optimized implementation
        └── z_score_norm__Original__.cpp   # Original baseline implementation
```

## Optimization Approaches

### OpenMP Implementation

The OpenMP implementation parallelizes the z-score normalization process using multi-threading. This approach distributes the workload across multiple CPU cores, significantly reducing computation time for large datasets.

Key optimizations include:
- Parallel calculation of sum and sum of squares
- Parallel normalization of data points
- Optimized memory access patterns

For detailed performance analysis, refer to `OpenMP/report.pdf`.

### SIMD Implementation

The SIMD implementation uses vectorized instructions that allow processing multiple data elements in a single CPU instruction. This approach leverages modern CPU architectures to perform the same operation on multiple data points simultaneously.

Key optimizations include:
- Vectorized mean calculation
- Vectorized standard deviation calculation
- Aligned memory access for better performance
- SIMD intrinsics utilization

For detailed performance analysis, refer to `SIMD/report.pdf`.

## Performance Results

Both optimization approaches show significant speedups compared to the original implementation:

### OpenMP Results
- The OpenMP implementation achieves up to 6.45x speedup on multi-core systems.
- Scaling improves with larger dataset sizes.
- Performance analysis showed the original implementation averaged 41.79 seconds versus 6.48 seconds with OpenMP.

### SIMD Results
- The SIMD implementation achieves up to 5.16x speedup.
- Most efficient with aligned data and when processing large contiguous arrays.
- Performance analysis showed the original implementation averaged 23.10 seconds versus 4.47 seconds with the SIMD version.

Detailed performance comparisons and analysis are available in the respective Jupyter notebooks and reports.

## Building and Running

### Prerequisites
- C++ compiler with OpenMP support (GCC or Clang)
- CPU with SIMD x86 instruction support (SSE/AVX)
- Python with Jupyter (for running analysis notebooks)


## Benchmark Scripts

The performance measurements were obtained by running each implementation multiple times and calculating the average execution time. Below are the scripts used to perform these benchmarks:

### OpenMP Benchmark Script

```bash
#!/bin/bash
# Script to benchmark OpenMP implementation vs original

echo -e "-----\n\n\n\n\n ################### ORIGINAL ###################"
g++ ./OpenMP/src/z_score_norm_original.cpp -o original.exe
for i in {1..20}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./original.exe
done

echo -e "-----\n\n\n\n\n ################### OpenMP ###################"
g++ -fopenmp ./OpenMP/src/z_score_norm_openMP.cpp -o OpenMP.exe 
for i in {1..20}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./OpenMP.exe
done
```

### SIMD Benchmark Script

```bash
#!/bin/bash
# Script to benchmark SIMD implementation vs original

echo -e "-----\n\n\n\n\n ################### ORIGINAL ###################"
g++ ./SIMD/src/z_score_norm__Original__.cpp -o z_norm_og.exe
for i in {1..15}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./z_norm_og.exe
done

echo -e "-----\n\n\n\n\n ################### SIMD ###################"
g++ -mavx2 ./SIMD/src/z_score_norm__Optimized__.cpp -o z_opt.exe
for i in {1..15}
do
  echo -e "-----\n\n\n\nRun #$i"
  ./z_opt.exe
done
```

To run these scripts:
1. Save the script to a file (e.g., `benchmark_openmp.sh` or `benchmark_simd.sh`)
2. Make it executable: `chmod +x benchmark_openmp.sh`
3. Run it: `./benchmark_openmp.sh > openmp.log`

The execution time output will be saved to the log file for further analysis.

## Conclusion

This project demonstrates how computational performance for z-score normalization can be significantly improved through parallel processing and vectorization techniques. Both OpenMP and SIMD approaches offer substantial speedups, with the optimal choice depending on the specific hardware architecture and dataset characteristics.

## License

This project is licensed under the terms of the included LICENSE file.
