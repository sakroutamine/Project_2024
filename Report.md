# CSCE 435 Group project

## 0. Group number:

5

## 1. Group members:

1. Amine Sakrout
2. Amol Gupta
3. Quenton Hua
4. Mohsin Khan
5. Oreoluwa Ogunleye-Olawuyi

### 1a. Communication

We will be working together using Discord and text to ensure everyone is up to date and able to continue working on the assignment.

## 2. Project topic (e.g., parallel sorting algorithms)

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)

- Bitonic Sort (Amol Gupta): Bitonic Sort is a parallel sorting algorithm that works by creating a bitonic sequence (a sequence that first increases and then decreases). The algorithm recursively sorts the sequence by comparing and swapping elements to form the bitonic sequence, and then merges the sequence to produce a sorted list. To parallelize Bitonic Sort with OpenMPI, the dataset will be divided into smaller sub-sequences, each assigned to different processors. Each processor independently will then sort its sub-sequence into a bitonic sequence, and then the sequences will be merged in parallel to yield a fully sorted list.
- Sample Sort: This algorithm splits up the dataset into smaller sample sizes and sorts these smaller groups using something like merge or quick sort. This sorting can be parallelized using OpenMPI to speed up this process. Once these groups are sorted they are merged together to yeild a fully sorted list.
- Merge Sort (Quenton Hua): Merge Sort is a sorting algorithm that recursively divides a list into two halves, sorts each half, and merges the sorted halves to produce a sorted list. To parallelize Merge Sort with OpenMPI, the dataset will be divided into smaller chunks, each assigned to different processors. Each processor sorts its chunk independently which is then merged in parallel.
- Radix Sort (Mohsin Khan): This algorithm processes the dataset by sorting elements based on individual digits, starting from the least significant digit. The sorting of each digit can be parallelized using OpenMPI to speed up the process. After each digit is sorted, the data is redistributed across processes to ensure the correct order for the next digit. Once all digits have been processed, the result is a fully sorted list.

- Column Sort: is a parallel sorting algorithm for sorting multi-dimensional data in a matrix. The unsorted input array is split among multiple processors. Each processor sorts its own column of data. The matrix is transposed, so each processor now sorts its row. The matrix is transposed back to its original form, and the processor sort again, giving a fully sorted matrix.

### 2b. Pseudocode for each parallel algorithm

- For MPI programs, include MPI calls you will use to coordinate between processes
- Bitonic Sort-

```
function bitonicSort(arr, low, cnt, dir):
    if cnt > 1:
        k = cnt / 2
        bitonicSort(arr, low, k, 1)  // Sort in ascending order
        bitonicSort(arr, low + k, k, 0)  // Sort in descending order
        bitonicMerge(arr, low, cnt, dir)

function bitonicMerge(arr, low, cnt, dir):
    if cnt > 1:
        k = cnt / 2
        for i = low to low + k - 1:
            if (dir == (arr[i] > arr[i + k])):
                swap(arr[i], arr[i + k])
        bitonicMerge(arr, low, k, dir)
        bitonicMerge(arr, low + k, k, dir)

function parallelBitonicSort(arr):
    // Initialize MPI
    MPI_Init()
    rank = MPI_Comm_rank()
    size = MPI_Comm_size()

    // Divide the array into sub-sequences
    local_arr = divideArray(arr, size, rank)

    // Perform bitonic sort on local sub-sequence
    bitonicSort(local_arr, 0, length(local_arr), 1)

    // Gather sorted sub-sequences
    sorted_arr = MPI_Gather(local_arr, root=0)

    if rank == 0:
        // Merge sorted sub-sequences
        bitonicMerge(sorted_arr, 0, length(sorted_arr), 1)

    // Finalize MPI
    MPI_Finalize()

    return sorted_arr if rank == 0 else None

function bitonicSort(arr, low, cnt, dir):
    if cnt > 1:
        k = cnt / 2
        bitonicSort(arr, low, k, 1)  // Sort in ascending order
        bitonicSort(arr, low + k, k, 0)  // Sort in descending order
        bitonicMerge(arr, low, cnt, dir)

function bitonicMerge(arr, low, cnt, dir):
    if cnt > 1:
        k = cnt / 2
        for i = low to low + k - 1:
            if (dir == (arr[i] > arr[i + k])):
                swap(arr[i], arr[i + k])
        bitonicMerge(arr, low, k, dir)
        bitonicMerge(arr, low + k, k, dir)

function parallelBitonicSort(arr):
    // Initialize MPI
    MPI_Init()
    rank = MPI_Comm_rank()
    size = MPI_Comm_size()

    // Divide the array into sub-sequences
    local_arr = divideArray(arr, size, rank)

    // Perform bitonic sort on local sub-sequence
    bitonicSort(local_arr, 0, length(local_arr), 1)

    // Gather sorted sub-sequences
    sorted_arr = MPI_Gather(local_arr, root=0)

    if rank == 0:
        // Merge sorted sub-sequences
        bitonicMerge(sorted_arr, 0, length(sorted_arr), 1)

    // Finalize MPI
    MPI_Finalize()

    return sorted_arr if rank == 0 else None
```

- Sample Sort-

```
def parallel_sample_sort()
    initialize_parallel_environment()

    num_procs = get_num_processors()
    rank = get_processor_rank()

    MPI_Init(&argc, &argv);
    dataset = [provided data]
    gathered_samples = gather_samples_from_all_processors(dataset) #Split data
    if (rank == ROOT) {
        splitters = select_splitters(gathered_samples, num_procs) #Split collected data among all processors
    }
    exchanged_data = parallel_merge_sort(splitters) #Sort individual samples on parallel processors using merge sort
    local_sorted_data = merge_received_partitions(exchanged_data) #Combine results and sort

    global_sorted_data = gather_sorted_data(local_sorted_data) #Clean up returned data

return global_sorted_data

```

- Radix Sort-

```
Initialize MPI environment
MPI_Init()

Get rank of current process and the total number of processes
MPI_Comm_rank(MPI_COMM_WORLD, &rank)
MPI_Comm_size(MPI_COMM_WORLD, &size)

Master process initializes the data array if rank == 0 (master process)
if rank == 0 THEN
    Initialize the fullArray with N elements
end if

Broadcast the size of the data (N) to all processes using MPI_Bcast
MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD)

Divide the data into chunks for each process
subArraySize = N / size
Allocate memory for subArray of size subArraySize

Use MPI_Scatter to distribute parts of the full array from the master process to each process
MPI_Scatter(fullArray, subArraySize, MPI_INT, subArray, subArraySize, MPI_INT, 0, MPI_COMM_WORLD)

Find the maximum number in the local subArray to determine the number of digits
local_max = find_max(subArray)
Use MPI_Allreduce to find the global maximum value across all processes
MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD)

For each digit (starting from the least significant digit) until the most significant digit:
exp = 1
while global_max / exp > 0 do

    Perform counting sort on the current digit (exp) for the local subArray
    local_counting_sort(subArray, subArraySize, exp)

    Redistribute the data among processes based on the current digit using MPI_Alltoall
    MPI_Alltoall(local_data, subArraySize, MPI_INT, received_data, subArraySize, MPI_INT, MPI_COMM_WORLD)

    Update subArray with redistributed received_data for the next round
    subArray = received_data

    exp = exp * 10
end while

Optionally, gather all sorted subarrays at the master process using MPI_Gather
if rank == 0 THEN
    MPI_Gather(subArray, subArraySize, MPI_INT, fullArray, subArraySize, MPI_INT, 0, MPI_COMM_WORLD)
end if

Master process prints the fully sorted array
if rank == 0 THEN
    print sorted fullArray
end if

Finalize MPI environment
MPI_Finalize()
```

- Column Sort-

```
Input: A matrix M with n x n elements where n  is the total number of elements distributed among p processors
Output: A fully sorted matrix

  1. Distribute the matrix rows among processes using `MPI_Scatter`.
  2. Repeat these steps until sorted:
     - for each column in the local matrix:
       - Sort the column elements in ascending order.
     - Transpose the local matrix (rows become columns and columns become rows).
     - for each row in the local submatrix:
       - Sort the row elements in ascending order.
     - Use `MPI_Alltoall` to exchange rows among processes.
     - After each exchange, each process will have a subset of rows in the correct order.
     - Transpose the local submatrix again to get the original form.
     - for each column in the local submatrix:
       - Sort the column elements in ascending order.
  3. Gather all submatrices from processes into the root process using `MPI_Gather`.
  4. If the current process is the root process:
     - Print the fully sorted matrix.
  5. Finalize the MPI environment using `MPI_Finalize`.
```

- Merge Sort-

```
function parallelMerge(subArray, subArraySize, rank, size):
       Perform parallel merging in a tree
       step = 1
       while step < size do
           if rank % (2 * step) == 0 then
               // Check if the neighboring process (rank + step) exists
               if rank + step < size then
                      // Receive the sorted sub-array from the neighboring process
                      MPI_Recv(receive_buffer, subArraySize, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)

                      // Merge the received sub-array with the local sub-array
                      mergedArray = merge(subArray, receive_buffer)

                      // Update local sub-array with the merged result
                      subArray = mergedArray
                      subArraySize = subArraySize * 2
               end if
           else
               // Send the local sorted sub-array to the neighboring process (rank - step)
               MPI_Send(subArray, subArraySize, MPI_INT, rank - step, 0, MPI_COMM_WORLD)
               // Exit the loop once the array is sent
               break
           end if
           // Move to the next step in the merging process
           step = step * 2
       end while
```

- Column Sort:

### 2c. Evaluation plan - what and how will you measure and compare

- Input sizes, Input types: We will use various input sizes to evaluate the performance of each parallel sorting algorithm. The input sizes will follow powers of two: `2^16, 2^18, 2^20, 2^22, 2^24, 2^26, 2^28`.
- Strong scaling (same problem size, increase number of processors/nodes): We will measure **strong scaling** by fixing the input size and increasing the number of processors. We will evaluate how the runtime changes as we increase the number of MPI processes from `2, 4, 8, 16, 32, 64, 128, 256, 512, to 1024` for the same input sizes. This will help us determine how well the algorithms use additional processors. Furthermore, we will test this logic on four different kinds of input: sorted, random, reverse sorted, 1% permutation.
- Weak scaling (increase problem size, increase number of processors): We will increase the problem size and the number of processors proportionally, aiming to keep the workload per processor constant. By comparing the runtimes as we increase both input size and number of processors, we can evaluate the algorithms' ability to handle larger datasets without increasing the per-processor workload. Furthermore, we will test this logic on four different kinds of input: sorted, random, reverse sorted, 1% permutation.

### 3a. Caliper instrumentation

We are using 2^16 or 65536 values that make up the total array length along with using 32 processors.

- Merge Sort Calltree:

```
└─ 9.446 main
   ├─ 8.979 MPI_Comm_dup
   ├─ 0.000 MPI_Finalize
   ├─ 0.000 MPI_Finalized
   ├─ 0.000 MPI_Initialized
   ├─ 0.390 comm
   │  ├─ 0.164 comm_large
   │  │  ├─ 0.023 MPI_Gather
   │  │  └─ 0.140 MPI_Scatter
   │  └─ 0.227 comm_small
   │     └─ 0.226 MPI_Barrier
   ├─ 0.021 comp
   │  ├─ 0.021 comp_large
   │  └─ 0.000 comp_small
   ├─ 0.054 correctness_check
   │  ├─ 0.047 MPI_Bcast
   │  └─ 0.007 MPI_Gather
   └─ 0.000 data_init_runtime
```

- Bitonic sort Calltree:

```
7.322 main
├─ 0.000 MPI_Init
└─ 6.781 main
   ├─ 6.518 MPI_Comm_dup
   ├─ 0.000 MPI_Finalize
   ├─ 0.000 MPI_Finalized
   ├─ 0.000 MPI_Initialized
   ├─ 0.090 comm
   │  ├─ 0.074 MPI_Barrier
   │  ├─ 0.010 MPI_Scatter
   │  └─ 0.005 comm_large
   │     └─ 0.005 MPI_Gather
   ├─ 0.172 comp
   │  └─ 0.172 comp_large
   ├─ 0.012 correctness_check
   └─ 0.091 data_init_runtime
```

- Radix sort Calltree:

```
0.366 main
├─ 0.000 MPI_Init
├─ 0.000 data_init_runtime
├─ 0.007 MPI_Bcast
├─ 0.014 comm
│  ├─ 0.002 comm_small
│  │  └─ 0.002 MPI_Scatter
│  ├─ 0.002 MPI_Allreduce
│  └─ 0.010 comm_large
│     ├─ 0.002 MPI_Alltoall
│     ├─ 0.001 MPI_Alltoallv
│     └─ 0.001 MPI_Gather
├─ 0.005 comp
│  └─ 0.000 comp_small
├─ 0.000 correctness_check
│  ├─ 0.000 MPI_Gather
│  └─ 0.000 MPI_Bcast
├─ 0.000 MPI_Finalize
├─ 0.000 MPI_Initialized
├─ 0.000 MPI_Finalized
└─ 0.001 MPI_Comm_dup
```

- Sample Sort:

```1.629 main
├─ 0.000 MPI_Init
├─ 0.000 data_init_X
├─ 0.069 comm
│  ├─ 0.053 comm_small
│  │  └─ 0.053 MPI_Bcast
│  └─ 0.016 comm_large
│     ├─ 0.016 MPI_Scatter
│     └─ 0.000 MPI_Gather
├─ 0.001 comp
│  └─ 0.001 comp_large
├─ 0.000 correctness_check
│  ├─ 0.000 MPI_Gather
│  └─ 0.000 MPI_Bcast
├─ 0.000 MPI_Finalize
├─ 0.000 MPI_Initialized
├─ 0.000 MPI_Finalized
└─ 0.001 MPI_Comm_dup
```

## 4. Performance evaluation

### 1. Merge Sort:

#### 4.1a

<img width="300" alt="image" src="https://github.com/user-attachments/assets/2c6d92a6-8928-456c-82d5-0f6fec4df471">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/27049478-1eac-4041-b7fd-154b59976f7a">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/22b86a94-a958-4d45-ae07-ff5517d6c49f">

These plot show the speedup x # of processors for main, comm, and comp_large. From these plots we can see that the smaller array sizes benefit the least with smaller speedups. The speedups within main actually tend to decrease with more processors, this is due to the communication overhead as seen in the comm graph. The speedup for comm exponentially decreases with increased processors, plateauing with higher processors. This would make sense as more processors would be used hence more communication to allocate parts of the array would be required. For the comp_large the speedup increases with more processors, this makes sense as the computation load decreases per processors as its distributed among more processors. The smaller array actually benefits the most for comp_large with increased processors, this could be due to the fact that it requires less merging and swapping for a smaller array.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/7a892919-121a-4f04-b10a-c6daabbafb83">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/ce2aabdf-7ff7-4a05-835b-66b8fb415d85">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/30f52c7f-25d7-4550-a61e-292242e4c047">

In these graphs, we observe the performance of the merge sort algorithm for different input types: perturbed, random, reverse, and sorted. The first two graphs show that random input consistently results in the worst performance, with a sharp increase in time as the processor count grows. Perturbed input behaves similarly but is slightly better than random. Sorted and reverse inputs, on the other hand, remain the most efficient, where their performance growth is more gradual as the number of processors increase. The third graph shows minimal variation between input types when handling smaller inputs, likely indicating that all types are processed quickly and similarly across all variations made to processor count. As the processor count grows, random input clearly performs the worst likely due to the nature of the merge sort algorithm which would require merging and swapping elements for elements that are not sorted increasing its overhead by the most amongst all input types.

#### 4.1c

<img width="300" alt="image" src="https://github.com/user-attachments/assets/edd91c53-604f-4522-a705-c630290ae8e2">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/c17860bd-39ee-412d-a1c9-df05c2b59c6d">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/32c14247-8bc0-4f03-936e-43acca71323a">

The plots show the average, minimum, and maximum time per rank using a merge sort algorithm. The maximum time initially spikes higher than both the average and minimum times which could indicate a communication bottlenecks among ranks. As the array size increases, the gap between the minimum and maximum times narrows, suggesting improved load balancing with larger inputs. However, the maximum time remains consistently higher, which points to certain ranks taking longer most likely due to communication overhead.

### 1. Bitonic Sort:

### 4.1a

<img width="300" alt="image" src="https://github.com/user-attachments/assets/dc84c46b-465b-4050-9adc-f380e1f95be2">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/1d98956d-75aa-4bbd-9583-a5db3bcd946f">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/e43cd140-9397-4b76-99af-51436622ac41">

Given above are plots that show speedup achieved as the number of processors increases for the main, communication (comm), and large computation (comp_large) tasks specifically for a bitonic sorting algorithm. The results indicate that smaller array sizes see the least benefit, showing smaller speedups overall. This is probably due to the fact that smaller array sizes are less parallelizable which translates to lesser speedup values. For the main task, speedup tends to decline as the processor count grows as highlighted in the comm plot. This lines up with our expectations as an increase in processor count would translate to an increase in communication overhead whcih would reduce speedup as the processor count increases. In contrast, the speedup for comp_large improves as more processors are introduced, since distributing the computational load reduces the work per processor. We can also observe that smaller arrays have the highest speedups which is most likely due to the fact that they are computationally less expensive than the larger ones.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/f882e03d-073a-4b6c-a722-0f7dfc79b1a9">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/e0267fcb-2cab-4b96-a7ed-05386dc17f21">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/81696fbc-c3be-42c0-ad90-69caf47afb4e">

The provided plots illustrate the performance of the bitonic sort algorithm for four types of input: sorted, random, reverse, and perturbed. The first two plots indicate that the average time increases as the number of processors grows. This can be attributed to the "main" task, where combining data from multiple processors results in increased processing time. Interestingly, for random input, the rate of increase is slightly steeper compared to the other input types. Similarly, in the "comm" plot, the time taken increases with the number of processors, likely due to greater communication overhead, which is represented by an upward slope. Once again, random input shows the highest average time values, while perturbed and reverse inputs perform slightly worse than sorted input. Lastly, the "comp_large" plot reveals that the average time levels off at 128 processors, with no significant improvement in performance beyond this point. This trend remains consistent across all input types.

#### 4.1c

<img width="300" alt="image" src="https://github.com/user-attachments/assets/f3a8688a-2987-43cd-9642-61ed071b787d">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/d5f7b59b-ad8b-4e23-8a8a-7b94a865181e">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/d9efc4ea-266d-411a-8b32-79cccf2acbf4">

These plots show the average, minimum, and maximum time per rank when using the bitonic sort algorithm. For smaller arrays, the maximum time is much higher than both the average and minimum times, suggesting possible communication bottlenecks between ranks. As the array size increases, the delta between the minimum and maximum times decrease, indicating better distributed performance with larger inputs.

### 1. Sample Sort

<img width="300" alt="image" src="4.1a-1.png">
<img width="300" alt="image" src="4.1a-2.png">
<img width="300" alt="image" src="4.1a-3.png">

As we can see in the above images there is a speedup that can be seen when increasing the number of processors depending on the size of the original input. From 2 to around 64 processors as the input size gets larger the sort gets magnitutdes slower but this slowdown greatly diminishes when we go above 200 processors as we parallelize more of the work and see a big speedup. We also see speedup in the comm chart as processor count grows since more processors will be increasing the communication overheard time and reducing overall communication speed. The comp_large shows the speedup hitting a plateu until we reach input size of 2^28 which strangely has a large jump up in speedup.

<img width="300" alt="image" src="4.1a-4.png">
<img width="300" alt="image" src="4.1a-5.png">
<img width="300" alt="image" src="4.1a-6.png">

The provided plots illustrate the performance of the sample sort algorithm for four types of input: sorted, random, reverse, and perturbed. The second  graph shows how average communication time increases as we add more processors, which can again be attributed to the increase in overhead required for the processors to communicate with each other. We do see comp large average time decreasing as we add processors, which could be a result of the extra processing power dedicated to the job.

<img width="300" alt="image" src="4.1a-7.png">
<img width="300" alt="image" src="4.1a-8.png">
<img width="300" alt="image" src="4.1a-9.png">

The provided plots show the average, minimum, and maximum time per rank when using sample sort. We can see the first 2 graphs have similar times but as the input size gets larger and larger there is more of a slowdown in the algorithm

### 1. Radix Sort:

#### 4.1a

<img width="300" alt="image" src="https://github.com/user-attachments/assets/5b4e0c1b-5b80-4d2e-86fb-024b3eed877c">

<img width="300" alt="image" src="image.png">

<img width="300" alt="image" src="image-2.png">

Main: The speedup plot for the main function shows a sharp rise as we increase the number of processors, but the growth plateaus quickly, especially for larger input sizes. The highest speedups are achieved for medium-sized inputs (65536 and 262144), while smaller arrays do not benefit as much. The larger array sizes (67108864 and beyond) show diminishing returns with a larger number of processors, likely due to communication overhead.
Comm: The communication plot reveals that communication time grows more significant as we increase the number of processors, especially for large arrays. This reflects the increased overhead of distributing and collecting data across processors. The performance plateaus as we add more processors, highlighting the growing cost of communication with higher processor counts.
Comp: Computation speedup shows a linear increase for medium-sized arrays (65536–1048576) as the number of processors increases. Larger arrays take advantage of more processors effectively, with speedup continuing even at higher processor counts. However, the computation speedup for smaller inputs does not benefit as much from higher processor counts due to the lesser computation involved.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/7329d89f-c86e-4c35-936e-03ff4120a709">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/c33a4aaa-50d9-4a0e-8d25-afc0816d9592">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/1cb9356a-02d9-49df-a6a0-edd673269b73">

In contrast to what we might expect, the graphs for main, comm, and comp show little to no significant difference between the various input types (perturbed, random, reverse, sorted) as the array size increases. This minimal variance suggests that the Radix Sort algorithm is robust across input types, treating each input type similarly. Radix Sort’s ability to handle these diverse input types uniformly is likely due to its non-comparative nature. Unlike comparison-based sorting algorithms, Radix Sort focuses on digit-by-digit processing, making it insensitive to whether the input is sorted, reverse-sorted, or randomly distributed. The performance is largely determined by the input size rather than the structure of the input data, as indicated by the close clustering of lines for different input types across the graphs. This could also imply that for large data sets, the inherent properties of the input have little impact on the overall runtime and communication cost, as Radix Sort’s time complexity is dependent on the length and width of the keys rather than their order.

#### 4.1c

<img width="300" alt="image" src="image-3.png">

<img width="300" alt="image" src="image-4.png">

<img width="300" alt="image" src="image-5.png">

These plots show the average, minimum, and maximum time per rank as the number of processors increases. The maximum time per rank spikes initially for smaller arrays, likely due to load imbalances, as some processors finish their tasks quicker than others.
As the array size increases, the gap between minimum and maximum times narrows, indicating better load balancing for larger input sizes. This suggests that as the data grows, the workload becomes more evenly distributed among processors.
However, the maximum time remains consistently higher, especially for larger processor counts, indicating that certain ranks are consistently taking longer to process data, likely due to communication or computation bottlenecks in specific processors.

Column Sort:

There are currently issues with generating the caliper files for the column sort implementation. However, we noticed that as the input size for column sort increases, the time to get the output file also increases. For example, for input size 32, the time taken was 58s. For input size 64, the time taken increased to 5 minutes and 3 seconds. Interestingly when input size was increased to 128, the time taken went back to 58s. I believe this is because of the constraints around column sort. Column sort only works for inputs N numbers, which are treated as an r × s matrix, where N = rs, s is a divisor of r, and r ≥ 2(s − 1)^2. We generate our inputs automatically, which means that we have to generate a matrix that matches those constraints. For input size 64, we still have two columns with 32 numbers each which would take longer to sort than input size 32, we have 2 columns with 16 numbers. When the input size increased to 128, our input generated had 4 columns with 32 elements so it was faster for the column to be sorted.
As the number of elements N increases, the processor has more elements to sort so the local sorting processes sortColumns takes longer because they operate on larger datasets. Also with more elements, the amount of data that needs to be communicated between processes increases.
By increasing the processors, the amount of data to be divided among the processes decreases which reduces what each processor has to handle.

Include detailed analysis of computation performance, communication performance.
Include figures and explanation of your analysis.

### 4a. Vary the following parameters

For input_size's:

- 2^16, 2^18, 2^20, 2^22, 2^24, 2^26, 2^28

For input_type's:

- Sorted, Random, Reverse sorted, 1%perturbed

MPI: num_procs:

- 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

This should result in 4x7x10=280 Caliper files for your MPI experiments.

### 4b. Hints for performance analysis

To automate running a set of experiments, parameterize your program.

- input_type: "Sorted" could generate a sorted input to pass into your algorithms
- algorithm: You can have a switch statement that calls the different algorithms and sets the Adiak variables accordingly
- num_procs: How many MPI ranks you are using

When your program works with these parameters, you can write a shell script
that will run a for loop over the parameters above (e.g., on 64 processors,
perform runs that invoke algorithm2 for Sorted, ReverseSorted, and Random data).

### 4c. You should measure the following performance metrics

- `Time`
  - Min time/rank
  - Max time/rank
  - Avg time/rank
  - Total time
  - Variance time/rank

## 5. Presentation


### 1. Merge Sort:

#### Strong Scaling Plots (input_size)
##### comp_large

<img width="300" alt="image" src="https://github.com/user-attachments/assets/ebaaf474-b4ec-4155-ad89-02a74631872f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/af8a403b-1160-4f94-ad82-417548d11fbc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2ad7a89f-5d90-4628-a798-12f17d77e688">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/68673690-aeac-4f25-b973-b9a93444fdee">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/85196589-f390-43a4-a026-67c3761af27d">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/9d7c9a60-3b41-484e-b477-5667114e8d8e">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/15edc4b0-eae8-4845-83a4-f22b91e18211">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a69d37d4-e76f-49ee-baa1-0891975c9db0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e9a1228c-7fbc-41aa-9ed1-deafdce1c1f2">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/25e01288-fc93-489d-91d6-835b7a6821db">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/09d41598-901b-4cb6-b31e-618d0d2e4744">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/8f88e066-cda2-46c7-a75b-638bf86c4bf2">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/34daead0-bd66-4026-a2a7-a5c54b0ddeb3">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/0241daa5-2517-4b91-9841-06b11e38ed56">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/527fccaa-0b84-4391-b05f-c75c1757ad3c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/eac3d14f-cc0c-429e-b417-226e33cbd263">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/4efbc17e-1475-4078-851d-9bb0de9600f7">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d4628c71-6439-4a40-b81d-fbdb92c9bcda">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e8438651-15ee-4fcf-b4dd-fbafa6340a51">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e9bcd946-5873-4b9a-908e-d4a0b5540fe3">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/42b80d09-2636-4961-affb-be1324902d55">

#### Strong Scaling Speedup Plots (input_type)
##### comp_large

<img width="300" alt="image" src="https://github.com/user-attachments/assets/980f12b5-6884-4e33-9381-4431e78ac5e0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d8339b80-bb9e-4ee2-b14d-21178044479c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/8a4298e0-32cd-4002-ac38-5d22e3e51bb3">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/ef68e907-143c-43f2-98a5-29c7795f5c60">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/fbbfcb57-c557-416f-92d6-bee9a56bb23a">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/7c96e85b-b958-4eba-abec-bd35f4f7f176">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2d956b2e-df07-465e-aebd-2a99e7431a6c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/5be63293-bba8-48fd-8ea7-b24443f4460f">

##### main

<img width="300" alt="image" src="https://github.com/user-attachments/assets/de7503d4-79a2-478f-8593-9b857e98227f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d4e131ec-45e8-40fb-9139-d0bd08b05090">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/5babfba5-b1c0-4b58-b3da-c77253d007ad">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c5b34a8d-b032-46b9-ae1b-7deed53397fe">

#### Weak Scaling Speedup Plots (input_type)

##### comp_large
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b01b8243-a488-4b2e-ad74-c76bddf45d7c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a0acf5b9-750d-41c3-bdb0-e04173aa40cb">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/010ca110-2845-4308-b07b-da709116ced0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c40e9dd9-e6c3-4100-9f52-cf1fbb3d8709">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/f5ca0316-ac4d-4f0c-adca-15576236b420">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/54c3e9e3-e035-4da1-9756-bc8cb1ae5763">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e628e5e4-62f9-4bc1-b40d-057b16487bcc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c8eb063b-3fbe-4237-8348-1d840f622292">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b56be18d-8e2c-44cd-89f5-950901e4f07c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/66a6bef5-25c3-4278-82f7-583e6622e4e2">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/6a21ce30-fe6b-41ba-8bbb-40938ae62e2e">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d1a03cc7-9787-4907-9153-76e825c91be8">

### 2. Bitonic Sort:

#### Strong Scaling Plots (input_size)
##### comp_large

<img width="300" alt="image" src="https://github.com/user-attachments/assets/00ff828e-3803-4507-994a-e44eb843e8d0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/8dae6f89-e5a1-4dfb-b540-ccc77ee34d90">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/445cb5d1-e081-4437-91d5-b898638728d8">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/72c4cafc-0479-4454-8504-7fcfbd06661e">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d9d000ff-60a1-4418-a9ae-793d9a2815f1">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d7de9dc8-8df3-41ee-89c7-fd8a36535560">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b1cbf501-cf8f-4147-9677-4fdfa53fd5dd">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/bf092beb-0bc4-478e-ad24-f13ff2a6cdc1">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/7b7a7615-449d-41c0-8047-0af8ec03c095">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/379fa069-f1de-4081-883c-abb031a0423d">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/079b5390-2eb9-4f47-94d6-e6603ab6fe2a">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2f81f996-0d90-4370-bee9-a683760728b6">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/060ef127-4c9b-4b26-8944-23befdbb4339">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/3d59aa08-ca92-404d-a15e-b4c35e81b9af">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/4b2dd275-9b15-4783-aaef-2b5b0afc3940">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a82b1f75-b0d2-476e-9fa3-d273e7ab17eb">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/cdbe09bf-c233-42e2-ab5d-daa5c3764c17">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/f7e96fe8-f494-4618-b2bd-e65bcc30e908">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/6fe9af55-f90d-4d71-8bed-c22d34747f40">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/965ddd11-d0c7-4560-bf21-220ed9c74d08">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/9f995275-4500-4a5d-ae6f-989cd3fbfe5b">

#### Strong Scaling Speedup Plots (input_type)
##### comp_large
<img width="300" alt="image" src="https://github.com/user-attachments/assets/63e4ead0-3d9f-4143-8601-f87f76589b12">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/0f70e185-10d2-4729-893c-12a27b858789">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/5c32feb0-a47f-47ff-87c0-b4ddcfad817c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e9a4d468-1ec8-4ef3-97a3-9d4842c1360f">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2085971d-b934-475d-8665-14c9fce10f8d">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b1a20872-92d9-4679-9ca7-6496c91510cc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/fed83037-6c03-40d9-bb0e-85870e2f8094">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a73aa3e1-a1e2-4997-b3bc-c0e205bbf905">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/92ebcf75-978e-4cc7-8458-76d3f416bc0f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/741d4bdf-55ba-45f0-b4aa-67050cca1acc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e6624e96-0b61-41d2-b8f6-cc317aaac15a">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/cca86888-e6f1-46de-87e7-770b40d05100">

#### Weak Scaling Speedup Plots (input_type)
##### comp_large
<img width="300" alt="image" src="https://github.com/user-attachments/assets/0931b097-5c8d-4901-87ac-c716631500f5">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/78a3919e-9aed-4c96-88e5-8b8703924fb4">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/898051c8-f4d2-43da-b652-2519d4a4422c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2a4ba305-e4b5-494d-8898-dd2f63a96e74">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/1e80f3fd-ae36-4bdc-9d1f-e0159682fbbd">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/50e5e2e2-f817-4726-886f-db18b8f61a9f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/2eddecc1-af69-4d84-8190-611e36cc62a0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/113b7871-ec69-4824-bfa4-3983d178f3e8">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a3fdfa0f-4ade-486e-88ae-582b85068fb0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c3894884-d045-4640-abac-18245b66b857">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/214e40e0-c867-4177-9002-eedfe9660c7d">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/aeb60fa6-c9cd-421b-8ed9-1aa7cf8f4a77">

### 1. Radix Sort:

#### Strong Scaling Plots (input_size)
##### comp

<img width="300" alt="image" src="https://github.com/user-attachments/assets/3f490ff6-d926-4309-a6de-f999fa062f04">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/dfdc00d7-abc1-4718-a924-578fe2415816">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/26918144-70cb-4fe2-96c0-cec9293eec4a">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e06d1888-a75d-4793-a4aa-c006bde9a9ef">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/bd60553b-05ac-4214-bff9-1eb0574203bd">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/90b860ce-d5e3-465c-a211-3b1813ec1f8a">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/907778c6-e941-43c2-9e01-0d449b2bf52a">


##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/0bb3246c-b2c8-40ba-a68c-1619a4efe5fe">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b88731e8-b701-4736-a16b-55ba12737332">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e75c9bc2-2e64-478f-a861-57c29aeb1895">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/243bc772-b494-4a41-8fc4-d88fa0e0cb23">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/abf9221c-cd50-46de-96b5-2246db50ef38">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/beb36310-2072-401f-a053-6e479a332d0f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/05f6e3ec-9c9f-4394-904b-53c052f3f385">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b3e346d5-f827-4f1b-a843-420f19bdef9f">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/1151a432-18fd-4100-a233-fb19c917e4e7">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/bbd59457-3afe-4752-bd3c-552b68d6bc36">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/f85d5c6a-02ff-4c0d-a96f-dbcb9f23608c">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/70b49b55-650c-4458-9de0-9f461328d807">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/154fa773-1c4c-4952-8e59-75ada7dc67c9">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/e8b165f2-309f-4d28-83a8-0d84d33be906">

#### Strong Scaling Speedup Plots (input_type)
##### comp

<img width="300" alt="image" src="https://github.com/user-attachments/assets/104757c9-814e-4147-a126-e44d101b2709">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/5fe4bb78-109f-4364-bb74-94ac39e872ac">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/3611f4b5-c0fe-448d-b0d7-109ea95f93d8">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/48572d6c-c9a5-46f6-8e3e-7ff6f445e1bd">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/8ad7d4a1-a1b5-46b1-b0ef-4c2263083ddf">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/c4d2f5e0-8418-47a5-9653-d6ab3d2bd7a1">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/45e862b2-982e-41ca-a2f1-d3f5d2daa0cc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/de021451-1564-46f9-b18b-0a3512f9d532">

##### main

<img width="300" alt="image" src="https://github.com/user-attachments/assets/bf679d32-34a6-4f32-b563-09c78ba9d735">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/93d6a96f-e086-4b03-a319-04c2af3d0c08">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/7863402b-58bf-437c-a152-87b4068e7fca">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/abde962e-4566-45d4-9068-9071ca789a81">

#### Weak Scaling Speedup Plots (input_type)

##### comp
<img width="300" alt="image" src="https://github.com/user-attachments/assets/d092ab37-7c6f-4db2-8576-be17e663c357">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/95f93aa8-91eb-4879-b7c3-d162028108ad">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/30c61d59-e149-410c-84fe-6705bc40d280">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/19ab9847-ec78-4f7e-a829-08fd5bebbd9f">

##### comm
<img width="300" alt="image" src="https://github.com/user-attachments/assets/a17454f9-c72d-456d-afe1-74790aab4d63">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/267d47f4-a8f8-4faf-a286-493ae47aa7bb">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/b0ee1be8-c9e6-42f5-87ee-cebbf325c8a0">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/9a0e08ee-5ccf-4330-98cb-df7fd6e70fc1">

##### main
<img width="300" alt="image" src="https://github.com/user-attachments/assets/21e2e3e1-4b37-4354-9b6b-83f7c98cd0dd">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/211f821b-989a-436c-86ad-1a58ba247dcf">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/5bb9acdf-008c-4d13-bf4a-6e1a85cfdddc">
<img width="300" alt="image" src="https://github.com/user-attachments/assets/cd892df7-c182-42b8-910e-1d93232f9739">


Plots for the presentation should be as follows:

- For each implementation:
  - For each of comp_large, comm, and main:
    - Strong scaling plots for each input_size with lines for input_type (7 plots - 4 lines each)
    - Strong scaling speedup plot for each input_type (4 plots)
    - Weak scaling plots for each input_type (4 plots)

Analyze these plots and choose a subset to present and explain in your presentation.

## 6. Final Report

Submit a zip named `TeamX.zip` where `X` is your team number. The zip should contain the following files:

- Algorithms: Directory of source code of your algorithms.
- Data: All `.cali` files used to generate the plots seperated by algorithm/implementation.
- Jupyter notebook: The Jupyter notebook(s) used to generate the plots for the report.
- Report.md
