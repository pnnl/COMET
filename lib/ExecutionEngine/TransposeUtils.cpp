
//===- TransposeUtils.cpp - Utility functions for Tensor transpose -----------------===//
//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// This file includes runtime functions for sparse tensor
// transpose operations.
//
//===----------------------------------------------------------------------===//

#include "comet/ExecutionEngine/RunnerUtils.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>

// Parallel sorting algorithms
#include <algorithm>

using namespace std;

enum SortingOption
{
  NO_SORT,
  SEQ_QSORT,
  PAR_QSORT,
  RADIX_BUCKET,
  PAR_RADIX,
  PARTIAL_BUCKET_COUNT,
  COUNT_RADIX,
  COUNT_QUICK
};

int selected_sort_type = SEQ_QSORT;

struct coo_t
{
  vector<int32_t> coords;
  double val;
};

struct bucket
{
  int left;
  int right;
};

//===----------------------------------------------------------------------===//
// Different sorting algorithms
//===----------------------------------------------------------------------===//

bool sort_compare_coords(struct coo_t p, struct coo_t q)
{
  // Get the values at given addresses
  struct coo_t l = p;
  struct coo_t r = q;

  for (int i = 0; i < l.coords.size(); ++i)
  {
    if (l.coords[i] < r.coords[i])
    {
      return true;
    }
    else if (l.coords[i] > r.coords[i])
    {
      return false;
    }
  }
  return false;
}

int qsort_compare_coords(const void *p, const void *q)
{
  // Get the values at given addresses
  struct coo_t l = *(struct coo_t *)p;
  struct coo_t r = *(struct coo_t *)q;

  for (int i = 0; i < l.coords.size(); ++i)
  {
    if (l.coords[i] < r.coords[i])
    {
      return -1;
    }
    else if (l.coords[i] > r.coords[i])
    {
      return 1;
    }
  }
  return 0;
}

bool sort_compare_coord_2(struct coo_t p, struct coo_t q)
{
  return (p.coords[2] < q.coords[2]);
}

bool sort_compare_coord_1(struct coo_t p, struct coo_t q)
{
  return (p.coords[1] < q.coords[1]);
}

bool sort_compare_coord_0(struct coo_t p, struct coo_t q)
{
  return (p.coords[0] < q.coords[0]);
}

// A function to swap two elements
void swap(struct coo_t *a, struct coo_t *b)
{
  struct coo_t t = *a;
  *a = *b;
  *b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot) to
left of pivot and all greater elements to right of pivot */
int partition(vector<struct coo_t> &ary, int low, int high, int mod)
{
  // pivot
  struct coo_t pivot = ary[high];
  // Index of smaller element and indicates the right position of pivot found so far
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++)
  {
    // If current element is smaller than the pivot
    if (ary[j].coords[mod] < pivot.coords[mod])
    {
      i++; // increment index of smaller element
      swap(&ary[i], &ary[j]);
    }
  }
  swap(&ary[i + 1], &ary[high]);
  return (i + 1);
}

/* The QuickSort function
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index,
mod --> the coords mode to sort on (0, 1, 2, 3, ..)
*/
void quick_sort(vector<struct coo_t> &ary, int low, int high, int mod)
{
  if (low < high)
  {
    /* pi is partitioning index, ary[p] is now at right place */
    int pi = partition(ary, low, high, mod);

    // Separately sort elements before
    // partition and after partition
    quick_sort(ary, low, pi - 1, mod);
    quick_sort(ary, pi + 1, high, mod);
  }
}

void parallel_radix(vector<struct coo_t> &ary, int num_dims)
{
  if (num_dims == 3)
  {
    //__gnu_parallel::
    // sort(ary.begin(), ary.end(), sort_compare_coord_2);
    quick_sort(ary, 0, ary.size() - 1, 3 - 1);
  }

  //__gnu_parallel::
  quick_sort(ary, 0, ary.size() - 1, 2 - 1);

  //__gnu_parallel::
  quick_sort(ary, 0, ary.size() - 1, 1 - 1);
}

void count_sort(vector<struct coo_t> &ary, int n, int mode)
{
  // count sort for mode m
  vector<coo_t> sorted_ary(n);
  int maxx = 0, i, j, k;
  for (i = 0; i < n; i++)
  {
    if (maxx < ary[i].coords[mode])
      maxx = ary[i].coords[mode];
  }
  // Create a count array to store count of individual
  // characters and initialize count array as 0
  int count[maxx + 1];
  memset(count, 0, sizeof(count));

  // Store count of each number
  for (i = 0; i < n; ++i)
    ++count[ary[i].coords[mode]];

  // Change count[i] so that count[i] now contains actual
  // position of this character in output array
  for (i = 1; i <= maxx; ++i)
  {
    count[i] += count[i - 1];
  }

  // Build the output character array
  for (i = n - 1; i >= 0; --i)
  {
    sorted_ary[count[ary[i].coords[mode]] - 1] = ary[i];
    --count[ary[i].coords[mode]];
  }

  // Copy the sorted array to arr
  for (i = 0; i < n; ++i)
    ary[i] = sorted_ary[i];
}

// count sort with buckets output
void count_sort_output_bucket(vector<struct coo_t> &ary, int n, int mode, vector<bucket> &buckets)
{
  // count sort for mode m
  vector<coo_t> sorted_ary(n);
  int maxx = 0, i, j, k;
  for (i = 0; i < n; i++)
  {
    if (maxx < ary[i].coords[mode])
      maxx = ary[i].coords[mode];
  }
  // Create a count array to store count of individual
  // characters and initialize count array as 0
  int count[maxx + 1], count2[maxx + 1];
  memset(count, 0, sizeof(count));

  // Store count of each number
  for (i = 0; i < n; ++i)
    ++count[ary[i].coords[mode]];

  // Change count[i] so that count[i] now contains actual
  // position of this character in output array
  for (i = 0; i <= maxx; ++i)
  {
    if (i == 0)
    {
      count2[i] = 0;
      continue;
    }
    count[i] += count[i - 1];
    count2[i] = count[i];
  }

  // Build the output character array
  for (i = 0; i < n; ++i)
  {
    sorted_ary[count[ary[i].coords[mode]] - 1] = ary[i];
    --count[ary[i].coords[mode]];
  }

  // Copy the sorted array to arr
  for (i = 0; i < n; ++i)
    ary[i] = sorted_ary[i];

  // build buckets for the following sorting within buckets
  for (i = 1; i <= maxx; ++i)
  {
    if (count2[i - 1] < count2[i])
    {
      struct bucket abucket
      {
        count2[i - 1], count2[i]
      };
      buckets.push_back(abucket);
    }
  }
}

void partial_bucket_count(vector<struct coo_t> &ary, int n)
{
  // count sort for second digit
  vector<coo_t> sorted_ary(n);
  int maxx = 0, i, j, k;
  for (i = 0; i < n; i++)
  {
    if (maxx < ary[i].coords[1])
      maxx = ary[i].coords[1];
  }
  // Create a count array to store count of individual
  // characters and initialize count array as 0
  int count[maxx + 1];
  memset(count, 0, sizeof(count));

  // Store count of each character
  for (i = 0; i < n; ++i)
    ++count[ary[i].coords[1]];

  // Change count[i] so that count[i] now contains actual
  // position of this character in output array
  for (i = 1; i <= maxx; ++i)
    count[i] += count[i - 1];

  // Build the output character array
  for (i = 0; i < n; ++i)
  {
    sorted_ary[count[ary[i].coords[1]] - 1] = ary[i];
    --count[ary[i].coords[1]];
  }

  // assign the sorted array into buckets by bucket sorting
  vector<coo_t> buckets[n + 1];
  for (i = 0; i < n; ++i)
  {
    buckets[sorted_ary[i].coords[0]].push_back(sorted_ary[i]);
  }

  // dump buckets in sequence back to ary
  j = 0;
  k = 0;
  for (i = 0; i < n + 1; ++i)
  {
    if (!buckets[i].empty())
    {
      for (j = 0; j < buckets[i].size(); ++j)
      {
        ary[k] = buckets[i][j];
        k++;
      }
    }
  }
}

void radix_bucket(vector<struct coo_t> &ary, int n)
{
  int bucket_cnt[10];
  int i, j, k, r, NOP = 0 /*#digits*/, divisor = 1, maxx = 0, pass1, pass2;
  struct coo_t **bucket = new struct coo_t *[10];

  for (i = 0; i < 10; i++)
  {
    bucket[i] = new struct coo_t[n];
  }

  for (pass1 = 1; pass1 >= 0; pass1--)
  {
    // find the max val
    maxx = 0;
    for (i = 0; i < n; i++)
    {
      if (maxx < ary[i].coords[pass1])
        maxx = ary[i].coords[pass1];
    }

    // digists
    NOP = 0;
    while (maxx != 0)
    {
      maxx = maxx / 10;
      ++NOP;
    }

    divisor = 1;
    for (pass2 = 0; pass2 < NOP; pass2++)
    {
      for (i = 0; i < 10; i++)
      {
        bucket_cnt[i] = 0;
      }
      for (i = 0; i < n; i++)
      {
        r = (ary[i].coords[pass1] / divisor) % 10;
        bucket[r][bucket_cnt[r]] = ary[i];
        bucket_cnt[r] += 1;
      }
      i = 0;
      for (k = 0; k < 10; k++)
      {
        for (j = 0; j < bucket_cnt[k]; j++)
        {
          ary[i] = bucket[k][j];
          i++;
        }
      }
      divisor *= 10;
    }
  }

  // deallocate array
  for (i = 0; i < 10; i++)
  {
    delete[] bucket[i];
  }
  delete[] bucket;
}

// generate buckets for next sorting based on mode in question
void generate_buckets(vector<coo_t> &coo_ts, int mode, vector<bucket> &buckets)
{
  int count = 1;
  vector<int> counts;
  counts.push_back(0);
  for (int i = 1; i < coo_ts.size(); i++)
  {
    if (coo_ts[i].coords[mode] == coo_ts[i - 1].coords[mode])
    {
      count++;
    }
    else
    {
      counts.push_back(count);
      count++;
    }
  }
  counts.push_back(count);

  // merge previous bucket into new counts
  for (int i = 0; i < buckets.size(); i++)
  {
    counts.push_back(buckets[i].right);
  }
  sort(counts.begin(), counts.end());
  //__gnu_parallel::sort(counts.begin(), counts.end());
  buckets.clear();
  buckets.shrink_to_fit();

  for (int i = 1; i < counts.size(); i++)
  {
    if (counts[i - 1] < counts[i])
    {
      // if (bucket)
      struct bucket abucket
      {
        counts[i - 1], counts[i]
      };
      buckets.push_back(abucket);
    }
  }
}

// This hybrid sort function that combines count sort and quick sort,
// which use count sort for the 1st mode, and then quick sort on the
// remaining modes within buckets
void count_quick(vector<struct coo_t> &ary, int n, int num_dims)
{
  vector<bucket> buckets; // buckets generated by first sort
  count_sort(ary, n, 0);
  generate_buckets(ary, 0, buckets);
  for (int i = 0; i < buckets.size(); i++)
  {
    //__gnu_parallel::sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_1);
    sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_1);
  }
  if (num_dims > 2)
  {
    generate_buckets(ary, 1, buckets); // buckets generated by second sort
    for (int i = 0; i < buckets.size(); i++)
    {
      //__gnu_parallel::sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_2);
      sort(&ary[buckets[i].left], &ary[buckets[i].right], sort_compare_coord_2);
    }
  }
}

void count_radix(vector<struct coo_t> &ary, int n, int num_dims)
{
  //  the beginning of the file
  char *sd = getenv("SORT_DIMS");
  if (sd == NULL)
    assert(false && "set SORT_DIMS to be 1, 2, or 3 \n");

  int sort_dims = atoi(sd);

  if (num_dims == 2)
  {
    if (sort_dims == 1)
    {
      count_sort(ary, n, 0);
    }
    else if (sort_dims == 0)
    {
      ;
      // llvm::nulls();
    }
    else
    {
      assert(false && "ERROR: wrong dimensions for matrixes \n");
    }
  }

  if (num_dims == 3)
  {
    switch (sort_dims)
    {
    case 0:
      // neglect sorting
      break;
    case 1:
      count_sort(ary, n, 0);
      break;
    case 2:
      count_sort(ary, n, 1);
      count_sort(ary, n, 0);
      break;
    case 3:
      count_sort(ary, n, 2);
      count_sort(ary, n, 1);
      count_sort(ary, n, 0);
      break;
    default:
      assert(false && "ERROR: wrong dimensions for 3D tensors\n");
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

void transpose_sort(int sort_type, vector<coo_t> &coo_ts, int sz,
                    int num_dims, int output_permutation)
{
  switch (sort_type)
  {
  case SEQ_QSORT:
    std::qsort((void *)&coo_ts[0], sz, sizeof(struct coo_t), qsort_compare_coords);
    break;
  case PAR_QSORT:
    //__gnu_parallel::sort(coo_ts.begin(), coo_ts.end(), sort_compare_coords);
    sort(coo_ts.begin(), coo_ts.end(), sort_compare_coords);
    break;
  case RADIX_BUCKET:
    radix_bucket(coo_ts, sz);
    break;

  case PARTIAL_BUCKET_COUNT:
    // only works for 021
    if (num_dims == 3 && output_permutation == 21)
    {
      partial_bucket_count(coo_ts, sz);
    }
    else
    {
      assert(false && "Error: PARTIAL_BUCKET_COUNT only works for > 3D tensors and permutation occurring after the first digits\n");
    }
    break;
  case PAR_RADIX:
    parallel_radix(coo_ts, num_dims);
    break;
  case COUNT_RADIX:
    count_radix(coo_ts, sz, num_dims);
    break;
  case COUNT_QUICK:
    count_quick(coo_ts, sz, num_dims);
    break;
  case NO_SORT:
    break;
  }
}

template <typename T>
void transpose_2D(int32_t A1format, int32_t A2format, int A1pos_rank, void *A1pos_ptr,
                  int A1crd_rank, void *A1crd_ptr, int A2pos_rank, void *A2pos_ptr,
                  int A2crd_rank, void *A2crd_ptr, int Aval_rank, void *Aval_ptr,
                  int32_t B1format, int32_t B2format,
                  int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                  int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                  int Bval_rank, void *Bval_ptr,
                  int sizes_rank, void *sizes_ptr)
{
  // int i = 0;
  int num_dims = 2;
  std::string Aspformat;
  std::string Bspformat;

  auto *desc_A1pos = static_cast<StridedMemRefType<int64_t, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<int64_t, 1> *>(A1crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<int64_t, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<int64_t, 1> *>(A2crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  auto *desc_B1pos = static_cast<StridedMemRefType<int64_t, 1> *>(B1pos_ptr);
  auto *desc_B1crd = static_cast<StridedMemRefType<int64_t, 1> *>(B1crd_ptr);
  auto *desc_B2pos = static_cast<StridedMemRefType<int64_t, 1> *>(B2pos_ptr);
  auto *desc_B2crd = static_cast<StridedMemRefType<int64_t, 1> *>(B2crd_ptr);
  auto *desc_Bval = static_cast<StridedMemRefType<T, 1> *>(Bval_ptr);

  auto *desc_sizes = static_cast<StridedMemRefType<int64_t, 1> *>(sizes_ptr);

  int rowSize = desc_sizes->data[5];
  int colSize = desc_sizes->data[6];

  if ((A1format == Compressed_nonunique && A2format == singleton) || (A1format == singleton && A2format == Compressed_nonunique))
  {
    Aspformat.assign("COO");
  }
  else if ((A1format == Dense && A2format == Compressed_unique) || (A1format == Compressed_unique && A2format == Dense))
  {
    Aspformat.assign("CSR");
  }
  else
  {
    assert(false && "ERROR: At this time, only COO and CSR formats are supported for input for transpose\n");
  }

  if ((B1format == Compressed_nonunique && B2format == singleton) || (B1format == singleton && B2format == Compressed_nonunique))
  {
    Bspformat.assign("COO");
  }
  else if ((B1format == Dense && B2format == Compressed_unique) || (B1format == Compressed_unique && B2format == Dense))
  {
    Bspformat.assign("CSR");
  }
  else
  {
    assert(false && "ERROR: At this time, only COO and CSR formats are supported for output for transpose\n");
  }

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("COO") == 0)
  {

    int sz = desc_Aval->sizes[0];
    // vector of coordinates
    vector<coo_t> coo_ts(sz);

    // dimension, so we need to use indexing map for transpose
    //===----------------------------------------------------------------------===//
    // marshalling data for each sorting algorithms
    //===----------------------------------------------------------------------===//
    for (int i = 0; i < sz; ++i)
    {
      coo_ts[i].coords.push_back(desc_A2crd->data[i]);
      coo_ts[i].coords.push_back(desc_A1crd->data[i]);
      coo_ts[i].val = desc_Aval->data[i];
    }

    //===----------------------------------------------------------------------===//
    // Different sorting algorithm
    //===----------------------------------------------------------------------===//
    transpose_sort(selected_sort_type, coo_ts, sz, num_dims, 0);
    //===----------------------------------------------------------------------===//
    // push transposed coords to output tensors
    //===----------------------------------------------------------------------===//
    for (int i = 0; i < sz; ++i)
    {
      desc_B1crd->data[i] = coo_ts[i].coords[0];
      desc_B2crd->data[i] = coo_ts[i].coords[1];
      desc_Bval->data[i] = coo_ts[i].val;
    }
  }

  if (Aspformat.compare("CSR") == 0 && Bspformat.compare("CSR") == 0)
  {
    // SORT OR NO SORT?
    // currently you are setting COUNT_RADIX manually, it needs to be changed
    char *sOrNot = getenv("CSR_SORT_OR_NOT"); // 0 or 1
    int sort_not = 1;                         // default
    if (!(sOrNot == NULL))
    {
      sort_not = atoi(sOrNot); // read from env var.
    }

    if (sort_not == 0)
    {
      // It may be due to BRowSize and BColSize being not correct.
      // 1) not by sorting: only works for CSR/matrices
      // Atomic-based Transposition: retraverse the matrix from the transposed direction
      // B's row size == input's col size
      int BRowSize = desc_sizes->data[6];
      int BColSize = desc_sizes->data[5];
      int BNnz = desc_Aval->sizes[0];

      // B's col pos size == B's #rows + 1
      // BColPosSize = BRowSize+1;
      // BColCrdSize = BNNZ
      // BValSize = BNNZ
      desc_B2pos->sizes[0] = BRowSize + 1;
      int count = 0;
      desc_B2pos->data[0] = count;
      int i, j, k;
      for (k = 0; k < BRowSize; k++)
      {
        for (i = 0; i < BColSize; i++)
        {
          for (j = desc_A2pos->data[i]; j < desc_A2pos->data[i + 1]; j++)
          {
            if (desc_A2crd->data[j] == k)
            {
              desc_B2crd->data[count] = i;
              desc_Bval->data[count] = desc_Aval->data[j];
              count++;
            }
          }
        }
        desc_B2pos->data[k + 1] = count;
      }
    }

    if (sort_not == 1)
    {
      // 2) by sorting: work for both CSR/matrices and CSF/tensors
      // CSR -> COO -> Transpose -> Sort -> COO -> CSR
      //===----------------------------------------------------------------------===//
      // marshalling data
      //===----------------------------------------------------------------------===//
      // vector of coordinates
      int i, j, k;
      int BNnz = desc_Aval->sizes[0];
      vector<coo_t> coo_ts(BNnz);
      int pos = 0;
      int counter = 0;
      for (i = 0; i < BNnz; ++i)
      {
        coo_ts[i].coords.push_back(desc_A2crd->data[i]);
        coo_ts[i].val = desc_Aval->data[i];
        counter++;
        for (j = 0; j < desc_A2pos->sizes[0] - 1; j++)
        {
          if (counter > desc_A2pos->data[j] && counter <= desc_A2pos->data[j + 1])
          {
            pos = j;
          }
        }
        coo_ts[i].coords.push_back(pos);
      }
      // sort the first dim
      //===----------------------------------------------------------------------===//
      // Different sorting algorithm
      //===----------------------------------------------------------------------===//
      //
      setenv("SORT_DIMS", "1", 1);
      transpose_sort(selected_sort_type, coo_ts, BNnz, num_dims, 0);

      // push sorted data back to B
      int BRowSize = colSize;
      desc_B1pos->data[0] = BRowSize;
      desc_B1crd->data[0] = -1;

      desc_B2pos->sizes[0] = BRowSize + 1; // resize
      // push pos to B
      counter = 1;
      j = 1;
      for (i = 1; i < BNnz; i++)
      {
        if (coo_ts[i - 1].coords[0] != coo_ts[i].coords[0])
        {
          desc_B2pos->data[j] = counter;
          j++;
        }
        counter++;
      }
      desc_B2pos->data[j] = counter;
      // push crd to B
      for (i = 0; i < BNnz; i++)
      {
        desc_B2crd->data[i] = coo_ts[i].coords[1];
        desc_Bval->data[i] = coo_ts[i].val;
      }
    }
  }

  if (Aspformat.compare("CSR") == 0 && Bspformat.compare("COO") == 0)
  {
    // CSR -> COO -> Transpose -> Sort -> COO
    int i, j, k;
    int BNnz = desc_Aval->sizes[0];
    vector<coo_t> coo_ts(BNnz);
    int pos = 0;
    int counter = 0;
    for (i = 0; i < BNnz; ++i)
    {
      coo_ts[i].coords.push_back(desc_A2crd->data[i]);
      coo_ts[i].val = desc_Aval->data[i];
      counter++;
      for (j = 0; j < desc_A2pos->sizes[0] - 1; j++)
      {
        if (counter > desc_A2pos->data[j] && counter <= desc_A2pos->data[j + 1])
        {
          pos = j;
        }
      }
      coo_ts[i].coords.push_back(pos);
    }
    // sort the first dim
    //===----------------------------------------------------------------------===//
    // Different sorting algorithm
    //===----------------------------------------------------------------------===//
    //
    setenv("SORT_DIMS", "1", 1);
    transpose_sort(selected_sort_type, coo_ts, BNnz, num_dims, 0);
    //===----------------------------------------------------------------------===//
    // push transposed coords to output tensors
    //===----------------------------------------------------------------------===//
    for (i = 0; i < BNnz; ++i)
    {
      desc_B1crd->data[i] = coo_ts[i].coords[0];
      desc_B2crd->data[i] = coo_ts[i].coords[1];
      desc_Bval->data[i] = coo_ts[i].val;
    }
  }

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("CSR") == 0)
  {
    int sz = desc_Aval->sizes[0];
    // vector of coordinates
    vector<coo_t> coo_ts(sz);

    // dimension, so we need to use indexing map for transpose
    //===----------------------------------------------------------------------===//
    // marshalling data for each sorting algorithms
    //===----------------------------------------------------------------------===//
    for (int i = 0; i < sz; ++i)
    {
      coo_ts[i].coords.push_back(desc_A2crd->data[i]);
      coo_ts[i].coords.push_back(desc_A1crd->data[i]);
      coo_ts[i].val = desc_Aval->data[i];
    }

    //===----------------------------------------------------------------------===//
    // Different sorting algorithm
    //===----------------------------------------------------------------------===//
    transpose_sort(selected_sort_type, coo_ts, sz, num_dims, 0);

    // COO --> CSR
    int BRowSize = colSize;
    desc_B2pos->sizes[0] = BRowSize + 1; // resize
    // push pos to B
    int counter = 1;
    int j = 1;
    for (int i = 1; i < sz; i++)
    {
      if (coo_ts[i - 1].coords[0] != coo_ts[i].coords[0])
      {
        desc_B2pos->data[j] = counter;
        j++;
      }
      counter++;
    }
    desc_B2pos->data[j] = counter;
    // push crd to B
    for (int i = 0; i < sz; i++)
    {
      desc_B2crd->data[i] = coo_ts[i].coords[1];
      desc_Bval->data[i] = coo_ts[i].val;
    }
  }
}

template <typename T>
void transpose_3D(int32_t input_permutation, int32_t output_permutation,
                  int32_t A1format, int32_t A2format, int32_t A3format,
                  int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                  int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                  int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                  int Aval_rank, void *Aval_ptr,
                  int32_t B1format, int32_t B2format, int32_t B3format,
                  int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                  int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                  int B3pos_rank, void *B3pos_ptr, int B3crd_rank, void *B3crd_ptr,
                  int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{

  int i = 0;
  int num_dims = 3;

  std::string Aspformat;
  std::string Bspformat;

  if ((A1format == Compressed_nonunique && A2format == singleton && A3format == singleton) ||
      (A1format == singleton && A2format == Compressed_nonunique && A3format == singleton) ||
      (A1format == singleton && A2format == singleton && A3format == Compressed_nonunique))
  {
    Aspformat.assign("COO");
  }
  else if (A1format == Compressed_unique &&
           A2format == Compressed_unique &&
           A3format == Compressed_unique)
  {
    Aspformat.assign("CSF");
  }
  else
  {
    assert(false && "ERROR: At this time, only COO and CSF formats are supported for input tensor.\n");
  }
  if ((B1format == Compressed_nonunique && B2format == singleton && B3format == singleton) ||
      (B1format == singleton && B2format == Compressed_nonunique && B3format == singleton) ||
      (B1format == singleton && B2format == singleton && B3format == Compressed_nonunique))
  {
    Bspformat.assign("COO");
  }
  else if (B1format == Compressed_unique &&
           B2format == Compressed_unique &&
           B3format == Compressed_unique)
  {
    Bspformat.assign("CSF");
  }
  else
  {
    assert(false && "ERROR: At this time, only COO and CSF formats are supported for output tensor.\n");
  }

  auto *desc_A1pos = static_cast<StridedMemRefType<int64_t, 1> *>(A1pos_ptr);
  auto *desc_A1crd = static_cast<StridedMemRefType<int64_t, 1> *>(A1crd_ptr);
  auto *desc_A2pos = static_cast<StridedMemRefType<int64_t, 1> *>(A2pos_ptr);
  auto *desc_A2crd = static_cast<StridedMemRefType<int64_t, 1> *>(A2crd_ptr);
  auto *desc_A3pos = static_cast<StridedMemRefType<int64_t, 1> *>(A3pos_ptr);
  auto *desc_A3crd = static_cast<StridedMemRefType<int64_t, 1> *>(A3crd_ptr);
  auto *desc_Aval = static_cast<StridedMemRefType<T, 1> *>(Aval_ptr);

  auto *desc_B1pos = static_cast<StridedMemRefType<int64_t, 1> *>(B1pos_ptr);
  auto *desc_B1crd = static_cast<StridedMemRefType<int64_t, 1> *>(B1crd_ptr);
  auto *desc_B2pos = static_cast<StridedMemRefType<int64_t, 1> *>(B2pos_ptr);
  auto *desc_B2crd = static_cast<StridedMemRefType<int64_t, 1> *>(B2crd_ptr);
  auto *desc_B3pos = static_cast<StridedMemRefType<int64_t, 1> *>(B3pos_ptr);
  auto *desc_B3crd = static_cast<StridedMemRefType<int64_t, 1> *>(B3crd_ptr);
  auto *desc_Bval = static_cast<StridedMemRefType<T, 1> *>(Bval_ptr);

  if (Aspformat.compare("COO") == 0 && Bspformat.compare("COO") == 0)
  {
    int sz = desc_Aval->sizes[0];

    // vector of coordinates
    vector<coo_t> coo_ts(sz);

    //===----------------------------------------------------------------------===//
    // marshalling data for permutation for transpose
    //===----------------------------------------------------------------------===//
    // There are 5 cases: 021, 102, 120, 201, 210
    for (int i = 0; i < sz; ++i)
    {
      for (int j = num_dims - 1; j >= 0; j--)
      {
        int tmp = pow(10, j);
        int digist = (output_permutation / tmp) % 10;
        switch (digist)
        {
        case 0:
          coo_ts[i].coords.push_back(desc_A1crd->data[i]);
          break;

        case 1:
          coo_ts[i].coords.push_back(desc_A2crd->data[i]);
          break;

        case 2:
          coo_ts[i].coords.push_back(desc_A3crd->data[i]);
          break;

        default:;
          assert(false && "ERROR: incorrect output permutation\n");
        }
      }
      coo_ts[i].val = desc_Aval->data[i];
    }

    //===----------------------------------------------------------------------===//
    // Different sorting algorithm
    //===----------------------------------------------------------------------===//
    transpose_sort(selected_sort_type, coo_ts, sz, num_dims, output_permutation);

    //===----------------------------------------------------------------------===//
    // push transposed coords to output tensors
    //===----------------------------------------------------------------------===//
    for (i = 0; i < sz; ++i)
    {
      desc_B1crd->data[i] = coo_ts[i].coords[0];
      desc_B2crd->data[i] = coo_ts[i].coords[1];
      desc_B3crd->data[i] = coo_ts[i].coords[2];
      desc_Bval->data[i] = coo_ts[i].val;
    }
  }

  if (Aspformat.compare("CSF") == 0 && Bspformat.compare("CSF") == 0)
  {
    int sz = desc_Aval->sizes[0];

    // vector of coordinates
    vector<coo_t> coo_ts(sz);

    // initialization
    for (int i = 0; i < sz; ++i)
    {
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].coords.push_back(desc_A3crd->data[i]);
      coo_ts[i].val = desc_Aval->data[i];
    }

    // fix coords for A1 and A2
    int i = 0, j = 0, k = 0, l = 0;
    for (i = 0; i < desc_A2crd->sizes[0]; ++i)
    {
      for (j = desc_A3pos->data[i]; j < desc_A3pos->data[i + 1]; ++j)
      {
        coo_ts[k].coords[1] = desc_A2crd->data[i]; // for A2
        k++;
      }
    }
    k = 0;
    for (i = 0; i < desc_A2pos->sizes[0] - 1; ++i)
    {
      for (j = desc_A3pos->data[desc_A2pos->data[i]]; j < desc_A3pos->data[desc_A2pos->data[i + 1]]; ++j)
      {
        coo_ts[k].coords[0] = desc_A1crd->data[i]; // for A1
        k++;
      }
    }

    // permutate coords
    // There are 5 cases: 021, 102, 120, 201, 210
    vector<coo_t> perm_coo_ts(sz);
    for (int i = 0; i < sz; ++i)
    {
      for (int j = num_dims - 1; j >= 0; j--)
      {
        int tmp = pow(10, j);
        int digist = (output_permutation / tmp) % 10;
        switch (digist)
        {
        case 0:
          perm_coo_ts[i].coords.push_back(coo_ts[i].coords[0]);
          break;

        case 1:
          perm_coo_ts[i].coords.push_back(coo_ts[i].coords[1]);
          break;

        case 2:
          perm_coo_ts[i].coords.push_back(coo_ts[i].coords[2]);
          break;

        default:
          std::cout << "ERROR: incorrect output permutation" << std::endl;
        }
      }
      perm_coo_ts[i].val = coo_ts[i].val;
    }

    //===----------------------------------------------------------------------===//
    // Different sorting algorithm
    //===----------------------------------------------------------------------===//
    transpose_sort(selected_sort_type, perm_coo_ts, sz, num_dims, output_permutation);

    //===----------------------------------------------------------------------===//
    // Convert COO back to CSF
    //===----------------------------------------------------------------------===//
    // calculate B1crd, B2crd, B3crd
    int counter = 0;
    for (i = 0; i < sz - 1; i++)
    {
      desc_B1crd->data[counter] = perm_coo_ts[i].coords[0];
      if (perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
      {
        counter++;
      }
    }
    desc_B1crd->data[counter] = perm_coo_ts[i].coords[0];
    desc_B1crd->sizes[0] = counter + 1;

    counter = 0;
    for (i = 0; i < sz - 1; i++)
    {
      desc_B2crd->data[counter] = perm_coo_ts[i].coords[1];
      if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1])
      {
        counter++;
      }
    }
    desc_B2crd->data[counter] = perm_coo_ts[i].coords[1];
    desc_B2crd->sizes[0] = counter + 1;

    for (i = 0; i < sz; i++)
    {
      desc_B3crd->data[i] = perm_coo_ts[i].coords[2];
      desc_Bval->data[i] = perm_coo_ts[i].val;
    }
    desc_B3crd->sizes[0] = sz;
    desc_Bval->sizes[0] = sz;

    // calculate B1pos, B2pos, B3pos
    desc_B1pos->data[0] = 0;
    desc_B1pos->data[1] = desc_B1crd->sizes[0];
    desc_B1pos->sizes[0] = 2;

    desc_B2pos->data[0] = 0;
    counter = 1;
    j = 1;
    for (i = 0; i < sz - 1; ++i)
    {
      if (perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
      {
        desc_B2pos->data[j] = counter;
        j++;
      }
      if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
          perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
      {
        counter++;
      }
    }
    desc_B2pos->data[j] = counter;
    desc_B2pos->sizes[0] = j + 1;

    desc_B3pos->data[0] = 0;
    counter = 1;
    j = 1;
    for (i = 0; i < sz - 1; ++i)
    {
      if (perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
          perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
      {
        desc_B3pos->data[j] = counter;
        j++;
      }
      if (perm_coo_ts[i].coords[2] != perm_coo_ts[i + 1].coords[2] ||
          perm_coo_ts[i].coords[1] != perm_coo_ts[i + 1].coords[1] ||
          perm_coo_ts[i].coords[0] != perm_coo_ts[i + 1].coords[0])
      {
        counter++;
      }
    }
    desc_B3pos->data[j] = counter;
    desc_B3pos->sizes[0] = j + 1;
  }
}

// 2D tensors
extern "C" void transpose_2D_f32(int32_t A1format, int32_t A2format,
                                 int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                 int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                 int Aval_rank, void *Aval_ptr, int32_t B1format, int32_t B2format,
                                 int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                 int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_2D<float>(A1format, A2format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      Aval_rank, Aval_ptr, B1format, B2format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      Bval_rank, Bval_ptr, sizes_rank, sizes_ptr);
}

extern "C" void transpose_2D_f64(int32_t A1format, int32_t A2format,
                                 int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                 int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                 int Aval_rank, void *Aval_ptr, int32_t B1format, int32_t B2format,
                                 int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                 int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_2D<double>(A1format, A2format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       Aval_rank, Aval_ptr, B1format, B2format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       Bval_rank, Bval_ptr, sizes_rank, sizes_ptr);
}

// 3D tensors
extern "C" void transpose_3D_f32(int32_t input_permutation, int32_t output_permutation,
                                 int32_t A1format, int32_t A2format, int32_t A3format,
                                 int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                 int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                 int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                 int Aval_rank, void *Aval_ptr,
                                 int32_t B1format, int32_t B2format, int32_t B3format,
                                 int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                 int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                 int B3pos_rank, void *B3pos_ptr, int B3crd_rank, void *B3crd_ptr,
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<float>(input_permutation, output_permutation,
                      A1format, A2format, A3format,
                      A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                      A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                      A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                      Aval_rank, Aval_ptr, B1format, B2format, B3format,
                      B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                      B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                      B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                      Bval_rank, Bval_ptr, sizes_rank, sizes_ptr);
}

extern "C" void transpose_3D_f64(int32_t input_permutation, int32_t output_permutation,
                                 int32_t A1format, int32_t A2format, int32_t A3format,
                                 int A1pos_rank, void *A1pos_ptr, int A1crd_rank, void *A1crd_ptr,
                                 int A2pos_rank, void *A2pos_ptr, int A2crd_rank, void *A2crd_ptr,
                                 int A3pos_rank, void *A3pos_ptr, int A3crd_rank, void *A3crd_ptr,
                                 int Aval_rank, void *Aval_ptr,
                                 int32_t B1format, int32_t B2format, int32_t B3format,
                                 int B1pos_rank, void *B1pos_ptr, int B1crd_rank, void *B1crd_ptr,
                                 int B2pos_rank, void *B2pos_ptr, int B2crd_rank, void *B2crd_ptr,
                                 int B3pos_rank, void *B3pos_ptr, int B3crd_rank, void *B3crd_ptr,
                                 int Bval_rank, void *Bval_ptr, int sizes_rank, void *sizes_ptr)
{
  transpose_3D<double>(input_permutation, output_permutation,
                       A1format, A2format, A3format,
                       A1pos_rank, A1pos_ptr, A1crd_rank, A1crd_ptr,
                       A2pos_rank, A2pos_ptr, A2crd_rank, A2crd_ptr,
                       A3pos_rank, A3pos_ptr, A3crd_rank, A3crd_ptr,
                       Aval_rank, Aval_ptr, B1format, B2format, B3format,
                       B1pos_rank, B1pos_ptr, B1crd_rank, B1crd_ptr,
                       B2pos_rank, B2pos_ptr, B2crd_rank, B2crd_ptr,
                       B3pos_rank, B3pos_ptr, B3crd_rank, B3crd_ptr,
                       Bval_rank, Bval_ptr, sizes_rank, sizes_ptr);
}
