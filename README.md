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
- The OpenMP implementation achieves up to X times speedup on multi-core systems.
- Scaling improves with larger dataset sizes.

### SIMD Results
- The SIMD implementation achieves up to Y times speedup.
- Most efficient with aligned data and when processing large contiguous arrays.

Detailed performance comparisons and analysis are available in the respective Jupyter notebooks and reports.

## Building and Running

### Prerequisites
- C++ compiler with OpenMP support (GCC or Clang)
- CPU with SIMD instruction support (SSE/AVX)
- Python with Jupyter (for running analysis notebooks)

### Compiling OpenMP Version
```bash
g++ -fopenmp -O3 -o z_score_openmp OpenMP/src/z_score_norm_openMP.cpp
```

### Compiling SIMD Version
```bash
g++ -mavx2 -O3 -o z_score_simd SIMD/src/z_score_norm__Optimized__.cpp
```

## Conclusion

This project demonstrates how computational performance for z-score normalization can be significantly improved through parallel processing and vectorization techniques. Both OpenMP and SIMD approaches offer substantial speedups, with the optimal choice depending on the specific hardware architecture and dataset characteristics.

## License

This project is licensed under the terms of the included LICENSE file.
