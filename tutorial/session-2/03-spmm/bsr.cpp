#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cusparse.h>
#include <assert.h>
#include <time.h>
#include <chrono>
#include <iostream>

const unsigned MAX_NUM_BLOCKS_X = 2147483647;
const unsigned MAX_NUM_BLOCKS_Y = 65535;
const unsigned MAX_NUM_BLOCKS_Z = 65535;
#define real double
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define ABS(a) (a > 0) ? a : -a
#ifdef NDEBUG
#define CUDA_CHECK(call) call
#else
#define CUDA_CHECK(call) \
do { \
    cudaError_t cudaStatus = call; \
    if (cudaStatus != cudaSuccess) { \
        err = 1; \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", cudaStatus, cudaGetErrorString(cudaStatus)); \
        exit(1); \
    } \
} while (0)
#endif
#ifdef NDEBUG
#define CUSPARSE_CHECK(call) call
#else 
#define CUSPARSE_CHECK(call) \
do { \
    cusparseStatus_t cusparseStatus = call; \
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) { \
        err = 1; \
        fprintf(stderr, "CUSPARSE Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d\n", cusparseStatus); \
        {\
        fprintf(stderr, "val %s\n", cusparseGetErrorString(cusparseStatus));\
        }\
        exit(1); \
    } \
} while (0)
#endif

#ifdef NDEBUG
#define CU_CHECK(call) call
#else 
#define CU_CHECK(call) \
do { \
    CUresult res = call; \
    if (res != CUDA_SUCCESS) { \
        err = 1; \
        fprintf(stderr, "CU Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d\n", res); \
        exit(1); \
    } \
} while (0)
#endif

void generate_spmatrix(size_t rowsA, size_t colsA, int** rowPtrA, int** colIndexA, real** valA, size_t bSizeA) {
    int* lPtrA = (int*)malloc((rowsA/bSizeA + 1) * sizeof(size_t));
    int* lIndexA = (int*)malloc(rowsA/bSizeA* colsA/bSizeA * sizeof(size_t));
    real* lvalA = (real*)malloc(rowsA * colsA * sizeof(size_t));
    lPtrA[0] = 0;
    
    for(size_t i = 1; i < rowsA/bSizeA+1; i++)
    {
        lPtrA[i] = lPtrA[i-1] + colsA/bSizeA;
        
        for(size_t j = 0; j < colsA/bSizeA; j++)
        {
            lIndexA[(i-1) * colsA/bSizeA +  j] = j;

            for(size_t k = 0; k < bSizeA * bSizeA; k++)
            {
                // printf("Col: %ld: %ld\n", ((i-1) * colsA/bSizeA + j) * bSizeA * bSizeA + k, k);
                lvalA[((i-1) * colsA/bSizeA + j) * bSizeA * bSizeA + k] = k;
                
                // printf("IDX2C Col: %ld\n", ((i-1) * colsA/bSizeA + j) * bSizeA * bSizeA + IDX2C(k/bSizeA, k%bSizeA, bSizeA));
                // lvalA[((i-1) * colsA/bSizeA + j) * bSizeA * bSizeA + IDX2C(k/bSizeA, k%bSizeA, bSizeA)] = k ;
                
            }
        }
    }

    *rowPtrA = lPtrA;
    *colIndexA = lIndexA;
    *valA = lvalA;
}

void generateDensematrix(size_t rowsA, size_t colsA, real* vals)
{
    // real* init = (real*)malloc(rowsA* colsA * sizeof(real));

    for(size_t i = 0; i < rowsA; i++)
    {
        for(size_t j = 0; j < colsA; j++)
        {
            vals[i* colsA + j] = i+j;
        }

    }
    // *vals = init;
}

void bcsrToDense(size_t rowsA, size_t colsA, real** denseA, int* rowPtrA, int* colIndexA, real* valA, size_t bSizeA)
{
    real* A = (real*)malloc(rowsA * colsA * sizeof(real));
    memset(A, 0, rowsA * colsA * sizeof(real));
    for(size_t i = 0; i < rowsA/bSizeA; i++)
    {
        for(size_t j = rowPtrA[i]; j < rowPtrA[i+1]; j++)
        {
            // size_t base = colIndexA[j]
            for(size_t ii = 0; ii < bSizeA; ii++)
            {
                for(size_t jj = 0; jj < bSizeA; jj++)
                {
                    // printf("JJ\n");
                    // printf("Aindex: %ld\n", (i * bSizeA + ii) * colsA + (j* bSizeA * bSizeA ) +jj);
                    // A[IDX2C()]
                    A[(i * bSizeA + ii) * colsA + (colIndexA[j]* bSizeA) +jj  ] = valA[j * bSizeA * bSizeA + ii * bSizeA + jj];
                }
            }
        }
    }

    *denseA = A;
}

void csrToDense(size_t rowsA, size_t colsA, real** denseA, int* rowPtrA, int* colIndexA, real* valA)
{
    real* A = (real*)malloc(rowsA * colsA * sizeof(real));
    memset(A, 0, rowsA * colsA * sizeof(real));
    for(size_t i = 0; i < rowsA; i++)
    {
        for(size_t j = rowPtrA[i]; j < rowPtrA[i+1]; j++)
        {
            // printf("JJ\n");
            // printf("Aindex: %ld\n", (i * bSizeA + ii) * colsA + (j* bSizeA * bSizeA ) +jj);
            // A[IDX2C()]
            A[(i * colsA) +colIndexA[j]] = valA[j];
        }
    }

    *denseA = A;
}

void matmul_dense(real* A, real* B, real* C, size_t m, size_t k, size_t n)
{
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            // C[IDX2C(i, j, m)] = 0.0;
            C[i*n + j] = 0;
            for(size_t l = 0; l < k; l++)
            {
                // printf("%f * %f\n", A[IDX2C(i, l, m)], B[IDX2C(l, j, k)]);
                // C[IDX2C(i, j, m)] += A[IDX2C(i, l, m)] * B[IDX2C(l, j, k)];
                C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
        }
    }
}

void matmul_csr(int* Aptr, int* Aind, real* Aval, real* B, real* C, size_t m, size_t k, size_t n)
{
    memset(C, 0, m*n*sizeof(real));
    for(size_t i =0; i < m ; i++)
    {
        for(size_t j = Aptr[i]; j < Aptr[i+1]; j++)
        {
            for(size_t l = 0; l < n; l++)
            {
                C[i*n + l] += Aval[j] * B[Aind[j]*n + l]; 

            }
        }
    }
}

void matmul_bcsr(int* Aptr, int* Aind, real* Aval, size_t blockSize, real* B, real* C, size_t m, size_t k, size_t n)
{
    memset(C, 0, m*n*sizeof(real));
    for(size_t i =0; i < m ; i++)
    {
        for(size_t j = Aptr[i]; j < Aptr[i+1]; j++)
        {
            for(size_t ii= 0; ii< blockSize;ii++)
            {
                for(size_t jj= 0; jj< blockSize;jj++)
                {
                    for(size_t l = 0; l < n; l ++)
                    {
                        C[(i*blockSize + ii)*n + l] += Aval[j*blockSize*blockSize + ii * blockSize + jj] * B[(Aind[j] * blockSize + jj)*n +l];
                    }
                }

            }
        }
    }
}

void rowToColMajor(real* Arow, real* Acol, size_t m, size_t n)
{
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            Acol[IDX2C(i, j, m)] = Arow[i * n + j ] ;
        }
    }
}
//  rowsA/blockSize, colsC, colsA/blockSize, nnzb,
//                                         &alpha, descrA, devBsrValA, devBsrRowPtrA, devBsrColIndA, blockSize,
//                                         devDenseB, rowsB, &beta, devDenseC, rowsC


size_t validate_result_col_major(real* res, real* expected, size_t rows, size_t cols)
{
    size_t valid = 1;
    for(size_t i =0; i < rows; i++)
    {
        for(size_t j =0; j < cols; j++)
        {
            if( fabs(expected[i*cols + j] - res[IDX2C(i, j, rows)]) >  0.1  )
            {
                fprintf(stderr, "Error (%ld, %ld) :  %f. %f vs %f\n", i,j, fabs(expected[i*cols + j] - res[IDX2C(i, j, rows)]), expected[i*cols + j], res[IDX2C(i, j, rows)]);
                valid = 0;
                return valid;
                // break;
            }
        }
    }

    return valid;
}
size_t validate_result_row_major(real* res, real* expected, size_t rows, size_t cols)
{
    size_t valid = 1;
    for(size_t i =0; i < rows; i++)
    {
        for(size_t j =0; j < cols; j++)
        {
            if( fabs(expected[i*cols + j] -  res[i* cols + j]) >  0.1  )
            {
                fprintf(stderr, "Error (%ld, %ld) : %f. %f vs %f\n", i,j, fabs(expected[i*cols + j]  - res[i* cols + j]), expected[i*cols + j], res[i* cols + j]);
                valid = 0;
                return valid;
                
                // break;
            }
        }
    }

    return valid;
}
    // cuSparseRes = runCuSparse(reps, num_rows, num_cols, rowsB, num_b_cols, rowsC, colsC, blocksize, nnzb, row_ptrs, colIndices, vals, dense_b, dense_C);
// func(A.shape[0], A.shape[1], A.nnz//(A.blocksize[0] * A.blocksize[1]), A.blocksize[0], A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), A.data.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), B.shape[1], B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

size_t runCuSparse(size_t reps, size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, size_t rowsC, size_t colsC, size_t blockSize, size_t nnzb, int *bsrRowPtrA, int *bsrColIndA, real* bsrValA, real* denseB, real* denseC)
{
    size_t err = 0;
    cusparseHandle_t handle;
    // Allocate and initialize matrices A (in BSR format) and B (dense)
    int *devBsrRowPtrA = NULL, *devBsrColIndA = NULL;
    real *devBsrValA = NULL, *devDenseB = NULL, *devDenseC = NULL;
    real alpha = 1.0f, beta = 0.0f;
    double total_time = 0.0;
    double gflops = 0.0;
    // clock_t before = 0, after = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    size_t  bufferSize = 0;
    void* devBuffer = NULL;
    size_t total_size = (rowsA + 1) * sizeof(int) +  nnzb * sizeof(int) + nnzb * blockSize * blockSize * sizeof(real) + rowsB * colsB * sizeof(real) + rowsC * colsC * sizeof(real);
    size_t Aval_size = nnzb * blockSize * blockSize;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUDA_CHECK(cudaMalloc((void**)&devBsrRowPtrA, (rowsA + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&devBsrColIndA, nnzb * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&devBsrValA, Aval_size * sizeof(real)));
    CUDA_CHECK(cudaMalloc((void**)&devDenseB, rowsB * colsB * sizeof(real)));
    CUDA_CHECK(cudaMalloc((void**)&devDenseC, rowsC * colsC * sizeof(real)));
    // std::cout << "In GPU allocating for Aval: " << Aval_size * sizeof(real) << "  bytes" << std::endl; 
    // std::cout << "In GPU allocating for rowOffsets: " << (rowsA + 1) * sizeof(int) << "  bytes" << std::endl; 
    // std::cout << "In GPU allocating for colIndices: " << nnzb * sizeof(int) << "  bytes" << std::endl; 
    // std::cout << "In GPU allocating for C: " << rowsC * colsC * sizeof(real) << "  bytes" << std::endl; 
    // std::cout << "In GPU allocating for B: " << rowsB * colsB * sizeof(real) << "  bytes" << std::endl; 
    // std::cout << "Abrows: " << rowsA << "Abcols: " << colsA << "Brows: " << rowsB << "  Bcols: " << colsB << " nnzb: " << nnzb << "\n";   
    // std::cout << "Crows: " << rowsC << "Ccols: " << colsC  << "\n";   
    CUDA_CHECK(cudaMemcpy(devBsrRowPtrA, bsrRowPtrA, (rowsA + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devBsrColIndA, bsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devBsrValA, bsrValA, Aval_size * sizeof(real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devDenseB, denseB, rowsB * colsB * sizeof(real), cudaMemcpyHostToDevice));
    
    cusparseMatDescr_t descrA, descrB, descrC;
    // Initialize cuSPARSE
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrB));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrC));
    
    // Set BSR matrix properties
    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));


    CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemset(devDenseC, 0 ,rowsC * colsC * sizeof(real));
    
    // begin = std::chrono::steady_clock::now();
    for(size_t r = 0; r < reps; r++)
    {
        assert(nnzb!= 0);
        assert(devBsrRowPtrA!= NULL);
        assert(devBsrColIndA!= NULL);
        assert(devBsrValA!= NULL);
        // printf("rowsA: %d\n", rowsA);
        // printf("colsB: %d\n", colsB);
        // printf("colsA: %d\n", colsA);
        // printf("nnzb: %d\n", nnzb);
        // printf("blockSize: %d\n", blockSize);
        // printf("colsB: %d\n", colsB);
        // printf("rowsC: %d\n", rowsC);
        // Perform BSR matrix-dense matrix multiplication
        CUSPARSE_CHECK(cusparseDbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_TRANSPOSE, rowsA, colsB, colsA, nnzb,
                            &alpha, descrA, devBsrValA, devBsrRowPtrA, devBsrColIndA, blockSize,
                            devDenseB, colsB, &beta, devDenseC, rowsC));

    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    // end = std::chrono::steady_clock::now();
    // std::cout << "Duration " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    // total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()/(real)reps;
    // #ifdef BCSR
    // gflops = (2UL * nnzb * blockSize * blockSize * colsC * 1e-12)/(total_time*1e-3);
    // #else
    // gflops = (2UL * nnz * colsC * 1e-9)/(total_time*1e-3);
    // #endif
    // printf("%ld %ld %ld %ld %lf (GFLOPs) %lf (ms)\n", rowsC, colsC, blockSize, nnz, gflops, total_time);

    // Copy the result back to host memory
    CUDA_CHECK(cudaMemcpy(denseC, devDenseC, rowsC * colsC * sizeof(real), cudaMemcpyDeviceToHost));
    
exit:

    // size_t validRow = validate_result_row_major(denseC, expRes, rowsC, colsC);
    // assert(validRow);

    // Free allocated memory
    cudaFree(devBsrRowPtrA);
    cudaFree(devBsrColIndA);
    cudaFree(devBsrValA);
    cudaFree(devDenseB);
    cudaFree(devDenseC);
    
    
    // Destroy cuSPARSE handle and descriptors
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrB));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrC));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    return err;
}


extern "C" int run_bsr(int reps, int num_rows, int num_cols, int nnzb, int blocksize, int* row_ptrs, int* colIndices, real* vals, int num_b_cols, real* dense_b, real* dense_C) {
    
    int err = 0;


    // size_t reps = 100;


    size_t validCol = 1;
    size_t cuSparseRes = 0, tritonRes = 0, cudaRes;
    real *expRes = NULL, *denseA = NULL;
    
    // int rows;
    // int cols;
    // int nnz;
    // int blockSize;
    // std::vector<int> rowPtrs;
    // std::vector<int> colIndices;
    // std::vector<real> vals;

    // #ifdef BCSR
    // bcsrMatrix spm = load_mm_as_bsr(filename, blockSize);
    // size_t rowsA = (spm.rows + spm.blockSize -1)/spm.blockSize * spm.blockSize;
    // size_t colsA = (spm.cols + spm.blockSize -1)/spm.blockSize * spm.blockSize;
    // size_t rowsB = (spm.rows + spm.blockSize -1)/spm.blockSize * spm.blockSize;
    // size_t colsB = (N + spm.blockSize -1)/spm.blockSize * spm.blockSize;
    // #else 
    // csrMatrix spm = load_mm_as_csr(filename);
    // size_t rowsA = spm.rows;
    // size_t colsA = spm.cols;
    // size_t rowsB = spm.rows;
    // size_t colsB = N;
    
    // #endif
    size_t rowsC = num_rows * blocksize;
    size_t colsC = num_b_cols;
    size_t rowsB = num_cols * blocksize;


    // real* denseBColM = (real*)malloc(rowsB*colsB*sizeof(real));
    // real* denseBRowM = (real*)malloc(rowsB*colsB*sizeof(real));
    // // std::cout << "Generating Dense matrices" << std::endl;
    // // memset(denseBRowM, 1, rowsB*colsB * sizeof(float) );
    // generateDensematrix(rowsB, colsB, denseBRowM);
    // rowToColMajor(denseBRowM, denseBColM, rowsB, colsB);
    // // memcpy(denseBColM, denseBRowM,rowsB*colsB * sizeof(real) );

    // std::cout << "Allocating for denseC: "<< rowsC*colsC*sizeof(real)  <<" bytes" << std::endl;

    // real *denseC = (real*)calloc(rowsC*colsC, sizeof(real));
    // expRes = (real*)calloc(rowsC*colsC, sizeof(real));
    // std::cout << "NNZB: " << spm.nnzb << "\n";


    cuSparseRes = runCuSparse(reps, num_rows, num_cols, rowsB, num_b_cols, rowsC, colsC, blocksize, nnzb, row_ptrs, colIndices, vals, dense_b, dense_C);

    return cuSparseRes;
}
