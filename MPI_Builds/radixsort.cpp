#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define BASE 10

int find_max(int* arr, int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void counting_sort(int* arr, int size, int exp) {
    int output[size]; 
    int count[BASE] = {0};

    // Count occurrences of digits
    for (int i = 0; i < size; i++) {
        count[(arr[i] / exp) % BASE]++;
    }

    // Update count[] to contain positions of digits in the output array
    for (int i = 1; i < BASE; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = size - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % BASE] - 1] = arr[i];
        count[(arr[i] / exp) % BASE]--;
    }

    // Copy the output array back to arr[]
    for (int i = 0; i < size; i++) {
        arr[i] = output[i];
    }
}

void generate_data(int* data, int N, const std::string& input_type) {
    if (input_type == "random") {
        for (int i = 0; i < N; i++) {
            data[i] = rand() % 100000;
        }
    } else if (input_type == "sorted") {
        for (int i = 0; i < N; i++) {
            data[i] = i;
        }
    } else if (input_type == "reverse") {
        for (int i = 0; i < N; i++) {
            data[i] = N - i;
        }
    } else if (input_type == "perturbed") {
        for (int i = 0; i < N; i++) {
            data[i] = i;
        }
        int perturb_count = N / 100;
        for (int i = 0; i < perturb_count; i++) {
            int idx1 = rand() % N;
            int idx2 = rand() % N;
            std::swap(data[idx1], data[idx2]);
        }
    }
}

bool check_sorted_correctly(int* local_data, int local_size, int rank, int size, MPI_Comm comm) {
    for (int i = 1; i < local_size; i++) {
        if (local_data[i - 1] > local_data[i]) {
            return false;
        }
    }

    // Check that the last value of this process is <= first value of the next process
    if (rank != size - 1) {
        int local_last = local_data[local_size - 1];
        int next_first;
        MPI_Send(&local_last, 1, MPI_INT, rank + 1, 0, comm);
        MPI_Recv(&next_first, 1, MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        if (local_last > next_first) {
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 1) {
        if (rank == 0) {
            printf("Usage: %s <input_size_power_of_2> <input_type (random|sorted|reverse|perturbed)>\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    int N_power = atoi(argv[1]);      // Input size as power of 2 (e.g., 16 for 2^16)
    // std::string input_type = argv[2]; // Input type (random, sorted, reverse, perturbed)
    std::string input_type = "random";
    int N = pow(2, N_power);          // Calculate 2^N

    adiak::value("input_size", N);       
    adiak::value("input_type", input_type.c_str()); 

    int fullArray[N];  
    int subArray[N / size];  

    CALI_MARK_BEGIN("data_init_runtime");
    if (rank == 0) {
        generate_data(fullArray, N, input_type); 
    }
    CALI_MARK_END("data_init_runtime");

    MPI_Bcast(&N, 1, MPI_INT, 0, comm);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Scatter(fullArray, N / size, MPI_INT, subArray, N / size, MPI_INT, 0, comm);
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    int local_max = find_max(subArray, N / size);
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    int global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, comm);
    CALI_MARK_END("comm");

    for (int exp = 1; global_max / exp > 0; exp *= BASE) {
        CALI_CXX_MARK_LOOP_BEGIN(sort_loop, "Sorting loop");

        CALI_MARK_BEGIN("comp");
        counting_sort(subArray, N / size, exp);
        CALI_MARK_END("comp");

        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        int received_data[N];  
        MPI_Alltoall(subArray, N / size, MPI_INT, received_data, N / size, MPI_INT, comm);
        for (int i = 0; i < (N / size) * size; i++) {
            subArray[i] = received_data[i];
        }
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");

        CALI_CXX_MARK_LOOP_END(sort_loop);
    }

    if (rank == 0) {
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        MPI_Gather(subArray, N / size, MPI_INT, fullArray, N / size, MPI_INT, 0, comm);
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        printf("Sorting completed.\n");
    }

    CALI_MARK_BEGIN("correctness_check");
    bool sorted_correctly = check_sorted_correctly(subArray, N / size, rank, size, comm);
    bool global_sorted;
    MPI_Reduce(&sorted_correctly, &global_sorted, 1, MPI_C_BOOL, MPI_LAND, 0, comm);
    CALI_MARK_END("correctness_check");

    if (rank == 0) {
        if (global_sorted) {
            printf("Array is sorted correctly.\n");
        } else {
            printf("Array is NOT sorted correctly.\n");
        }
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("algorithm", "radix"); // The name of the algorithm you are using (e.g., "merge", "bitonic")
    adiak::value("programming_model", "mpi"); // e.g. "mpi"
    adiak::value("data_type", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("size_of_data_type", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("input_size", N); // The number of elements in input dataset (1000)
    adiak::value("input_type", input_type); // For sorting, this would be choices: ("Sorted", "ReverseSorted", "Random", "1_perc_perturbed")
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("scalability", "strong"); // The scalability of your algorithm. choices: ("strong", "weak")
    adiak::value("group_num", "5"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online and ai"); // Where you got the source code of your algorithm. choices: ("online", "ai", "handwritten").

    mgr.stop();
    mgr.flush();
    adiak::finalize();
    MPI_Finalize();

    return 0;
}
