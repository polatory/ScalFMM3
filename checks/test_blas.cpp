/* C source code is found in dgemm_example.c */

#define macro_min(x, y) (((x) < (y)) ? (x) : (y))

#include "mkl.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    m = 20000, k = 200, n = 10000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n",
           m, k, k, n);
    alpha = 1.0;
    beta = 0.0;

    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
           " performance \n\n");
    A = (double*)mkl_malloc(m * k * sizeof(double), 64);
    B = (double*)mkl_malloc(k * n * sizeof(double), 64);
    C = (double*)mkl_malloc(m * n * sizeof(double), 64);
    if(A == NULL || B == NULL || C == NULL)
    {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf(" Intializing matrix data \n\n");
    for(i = 0; i < (m * k); i++)
    {
        A[i] = 1.0;
    }

    for(i = 0; i < (k * n); i++)
    {
        B[i] = 1.0;
    }

    for(i = 0; i < (m * n); i++)
    {
        C[i] = 0.0;
    }

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");

    auto start = std::chrono::system_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count()
              << "s\n";

    printf("\n Computations completed.\n\n");

    printf(" Top left corner of matrix A: \n");
    for(i = 0; i < macro_min(m, 6); i++)
    {
        for(j = 0; j < macro_min(k, 6); j++)
        {
            printf("%12.0f", A[j + i * k]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix B: \n");
    for(i = 0; i < macro_min(k, 6); i++)
    {
        for(j = 0; j < macro_min(n, 6); j++)
        {
            printf("%12.0f", B[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Top left corner of matrix C: \n");
    for(i = 0; i < macro_min(m, 6); i++)
    {
        for(j = 0; j < macro_min(n, 6); j++)
        {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }

    printf("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf(" Example completed. \n\n");
    return 0;
}
