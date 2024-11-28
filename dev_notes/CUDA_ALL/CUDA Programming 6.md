# CUDA Programming 6

## CUDA Stream

对于 CUDA stream 我只做简单的概念整理，以后看到不要慌就行。一般来说，kernel 外部的并行不是 CUDA 编程时考虑的重点，因为我们需要尽量减少主机与设备之间的数据传输以及主机中的计算

- CUDA stream，CUDA 流

  一个 CUDA stream 指的是由主机发出的在一个设备中执行的 CUDA 操作序列。stream 既可以从主机端发出，也可以从设备端发出，但是 stream 各个操作次序是由主机端控制，按照主机发布的次序执行

  任何 CUDA 操作都存在于某个 CUDA stream 中，要么是 default stream（也称空流 null stream），要么是指定的非空流 (我称为 non-default stream)。即：没有明确指定 CUDA stream 的 CUDA 操作都在 default stream 中执行

  **同一个 CUDA 流中的 CUDA 操作在设备中是顺序执行的，故同一个 CUDA 流中的核函数也必须在设备中顺序执行**

- Create & launch non-default CUDA stream

  一旦显式创建了流，这个流就是非空的，可以由 CUDA runtime api 创建和销毁

  ```c++
  cudaStream_t stream_1;
  cudaStreamCreate(&stream1);
  cudaStreamDestroy(stream_1);
  ```

  为了实现 CUDA stream 并发，主机在向某个 CUDA stream 发布一系列命令后就会马上获得程序的控制权，不会等待 CUDA stream 中的命令在设备上执行完毕。这样就可以通过主机产生多个相互独立的 CUDA stream

  要让 kernel 使用该 non-default stream 需要在 launching 时进行指定

  ```c++
  my_kernel<<<N_grid, N_block, N_shared, stream_id>>>
  ```

  需要注意的是，如果你要指定 stream，你就必须要指定 shared memory 大小，即使你没有使用也需要填0

  ```c++
  my_kernel<<<N_grid, N_block, 0, stream_id>>>
  ```

- Sync non-default CUDA stream

  CUDA runtime api 提供了同步主机和 CUDA stream 的功能

  ```c++ 
  cudaError_t cudaStreamSynchronize(cudaStream_t stream);
  cudaError_t cudaStreamQuery(cudaStream_t stream);
  ```

  `cudaStreamSynchronize` 会强制阻塞主机，直到 CUDA stream 中的所有操作都执行完毕。`cudaStreamQuery` 不会阻塞主机，只是检查 CUDA stream 中的所有操作是否都执行完毕。若是，返回 `cudaSuccess`，否则返回 `cudaErrorNotReady`

- Simple example of default CUDA stream

  教材举了一个例子说明一个 default stream 已经能够完成简单的主机和设备计算的并行

  ```c++
  cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
  sum<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
  ... // cpu operations;
  cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
  ```

  前两行 `cudaMemcpy` 是同步的，也就是说主机会等待他们执行完毕，而第三行 launch kernel 过后主机不会等待 CUDA stream 完成，会继续运行 cpu operations

  当遇到最后一行的 `cudaMemcpy` 时，由于同一个 CUDA stream 的操作是顺序执行的，所以如果之前的 `sum` 没有运行完毕的话，仍然会先等待然后再运行

- Pipeline parallel 流水线并行

  教材用一个简单的图示表示经典的流水线并行

  <img src="CUDA Programming 6/image-20241120171656568.png" alt="image-20241120171656568" style="zoom:80%;" />

## CUDA Library

- CUDA ecosystem

  CUDA 有非常完整的生态和工具集，在掌握 CUDA 基础之后，再学习一些 CUDA library 的使用将会起到事半功倍的效果。可以把 CUDA ecosystem 分为几个大类：

  1. 数学库：[cuBLAS](https://developer.nvidia.com/cublas), cuFFT, cuSPARSE, cuRAND, cuSolver, cuTENSOR, ...

  2. 并行算法库：[Thrust](https://developer.nvidia.com/thrust), CUB, ...

     > Using Thrust, C++ developers can write just a few lines of code to perform GPU-accelerated sort, scan, transform, and reduction operations orders of magnitude faster than the latest multi-core CPUs. 

  3. 图像和视频库：nvJPEG, NPP, Video Codec SDK, Optical Flow SDK, ...

  4. GPU 通信库：[NCCL](https://developer.nvidia.com/nccl), NVSHMEM, ...

  5. 深度学习库：[cuDNN](https://developer.nvidia.com/cudnn), [TensorRT](https://developer.nvidia.com/tensorrt), ...

- What is Thrust

  > Thrust 是一个实现了众多基本并行算法的 C++ 模板库，类似于 C++ STL，该库自动包含在 CUDA toolkit 中。该模板库仅由一些头文件组成，在使用某个功能时，包含需要的头文件即可，并且所有函数都在命名空间 `thrust` 中定义

  教材用了一个 prefix sum 的例子简要介绍了 thrust 的用法

  ```c++
  #include <thrust/device_vector.h>
  #include <thrust/scan.h>
  #include <stdio.h>
  
  int main(void)
  {
      int N = 10;
      thrust::device_vector<int> x(N, 0);
      thrust::device_vector<int> y(N, 0);
      for (int i = 0; i < x.size(); ++i)
      {
          x[i] = i + 1;
      }
      thrust::inclusive_scan(x.begin(), x.end(), y.begin());
      for (int i = 0; i < y.size(); ++i)
      {
          std::cout << y[i] << " ";
      }
      int test = 0;
      std::cout << test + y[0] << std::endl;
      return 0;
  }
  ```

  其中使用的 `inclusive_scan` 就是我们理解的 prefix sum 操作，并且为了使用 device vector 还需要头文件 `thrust/device_vector.h`

  可以看到，我们能够使用 `std::cout` & host variable 与 device vector 进行方便的交互，那么 `y[i]` 到底是一个 device variable 还是一个 host variable？

  > In the Thrust library, `thrust::device_vector` is a container that resides on the GPU device memory but provides an abstraction that allows you to access it as if it were a typical host vector. When you access elements of a `thrust::device_vector` (like `y[i]`), **Thrust implicitly copies the data from the device to the host for you to read.**

  这说明 device vector 为我们进行了方便的管理，不需要进行大量的 Memcpy 操作，这也是当你使用大量 thrust 库时推荐的选择。教材还介绍了不使用 device vector 而直接使用原始的 device pointer 操作，这就需要像之前一样用 `cudaMalloc` 去分配内存，我这里就不做整理了

- What is cuBLAS? (basic linear algebra subprograms)

  > cuBLAS is a GPU-accelerated library provided by NVIDIA that implements the Basic Linear Algebra Subprograms (BLAS) on NVIDIA GPUs

  教材仅用一个矩阵乘法的例子来展示下 cuBLAS 的使用

  ```c++
  #include "error.cuh" 
  #include <stdio.h>
  #include <cublas_v2.h>
  
  void print_matrix(int R, int C, double* A, const char* name);
  
  int main(void)
  {
      int M = 2;
      int K = 3;
      int N = 2;
      int MK = M * K;
      int KN = K * N;
      int MN = M * N;
  
      double *h_A = (double*) malloc(sizeof(double) * MK);
      double *h_B = (double*) malloc(sizeof(double) * KN);
      double *h_C = (double*) malloc(sizeof(double) * MN);
      for (int i = 0; i < MK; i++)
      {
          h_A[i] = i;
      }
      print_matrix(M, K, h_A, "A");
      for (int i = 0; i < KN; i++)
      {
          h_B[i] = i;
      }
      print_matrix(K, N, h_B, "B");
      for (int i = 0; i < MN; i++)
      {
          h_C[i] = 0;
      }
  
      double *g_A, *g_B, *g_C;
      CHECK(cudaMalloc((void **)&g_A, sizeof(double) * MK))
      CHECK(cudaMalloc((void **)&g_B, sizeof(double) * KN))
      CHECK(cudaMalloc((void **)&g_C, sizeof(double) * MN))
  
      cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
      cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);
      cublasSetVector(MN, sizeof(double), h_C, 1, g_C, 1);
  
      cublasHandle_t handle;
      cublasCreate(&handle);
      double alpha = 1.0;
      double beta = 0.0;
      cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
          M, N, K, &alpha, g_A, M, g_B, K, &beta, g_C, M);
      cublasDestroy(handle);
  
      cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
      print_matrix(M, N, h_C, "C = A x B");
  
      free(h_A);
      free(h_B);
      free(h_C);
      CHECK(cudaFree(g_A))
      CHECK(cudaFree(g_B))
      CHECK(cudaFree(g_C))
      return 0;
  }
  // impl of print_matrix is not listed here
  ```

  这里我重点关注下 `cublasDgemm_v2` 这个 API，解读一下其参数的作用以及该 API 本身的作用

  ```c++
  CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle,
                                                       cublasOperation_t transa,
                                                       cublasOperation_t transb,
                                                       int m,
                                                       int n,
                                                       int k,
                                                       const double* alpha,
                                                       const double* A,
                                                       int lda,
                                                       const double* B,
                                                       int ldb,
                                                       const double* beta,
                                                       double* C,
                                                       int ldc);
  ```

  这种问题交给 GPT 是再合适不过了

  > `cublasDgemm_v2` function is used to perform a Double-precision GEneral Matrix Multiplication (DGEMM) operation.
  >
  > GEMM can be expressed as:
  > $$
  > C = \alpha \times (A\times B) + \beta \times C
  > $$
  > **Parameters:**
  >
  > - **`cublasHandle_t handle`**: This is a handle to the cuBLAS library context. It manages resources and configurations during cuBLAS function calls. Before calling `cublasDgemm_v2`, you need to initialize this handle using `cublasCreate`.
  > - **`cublasOperation_t transa`**: Specifies whether to transpose matrix A. Options are `CUBLAS_OP_N` for no transpose, `CUBLAS_OP_T` for transpose, or `CUBLAS_OP_C` for conjugate transpose.
  > - **`cublasOperation_t transb`**: Specifies whether to transpose matrix B, with the same options as `transa`.
  > - **`int m`**: The number of rows in matrices A and C.
  > - **`int n`**: The number of columns in matrices B and C.
  > - **`int k`**: The number of columns in matrix A and rows in matrix B.
  > - **`const double\* alpha`**: A pointer to scalar alpha. This scalar multiplies the matrix product A×B*A*×*B*.
  > - **`const double\* A`**: A pointer to the matrix A in device memory.
  > - **`int lda`**: Leading dimension of A. This typically equals m unless A is a submatrix in a larger array.
  > - **`const double\* B`**: A pointer to the matrix B in device memory.
  > - **`int ldb`**: Leading dimension of B. This generally equals k unless B is a submatrix within a larger array.
  > - **`const double\* beta`**: A pointer to scalar beta. This scalar multiplies the input matrix C.
  > - **`double\* C`**: A pointer to the output matrix C in device memory.
  > - **`int ldc`**: Leading dimension of C. This is usually m unless C is part of a larger array.

  比较陌生的是 leading dimension 这个概念

  > The leading dimension of a matrix is essentially how matrix elements are laid out in memory. It is the "stride" or the number of elements between successive rows (in a column-major storage) or columns (in a row-major storage) of the matrix. The leading dimension is particularly important when dealing with submatrices or when the matrix is part of a larger storage array.

  所以说 cuBLAS 就是以 column major 来存储 matrix

  除了 `cublasDgemm` API 以外还有几个 API 的功能简单提一下：

  1. `cublasSetVector` [link](https://docs.nvidia.com/cuda/cublas/#cublassetvector)，功能是将 CPU 的 vector 数据 copy 到 GPU 上
  2. `cublasGetVector` [link](https://docs.nvidia.com/cuda/cublas/#cublasgetvector)，功能是将 GPU 数据 copy 到 CPU 上

  最后需要指出的是，在编译需要使用 cuBLAS 的 CUDA 程序时，需要 load 链接库

  ```shell
  nvcc -arch=sm_75 -lcublas cublas_gemm.cu
  ```

## Question

- How do I know what function is sync (blocking) and what function is async? e.g. cudaMemcpy is sync, but launching kernel is async

- What do these 2 macro `CUBLASAPI & CUBLASWINAPI` mean? 

  > **`CUBLASAPI` and `CUBLASWINAPI`**: These are platform-specific macros related to function calling conventions. They ensure compatibility and proper handling of the function on different systems. You typically don't need to worry about these when using the function.

- Is cuBLAS column major or raw major?

  column major