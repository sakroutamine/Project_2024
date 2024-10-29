#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <limits.h>
#include <string>
#include <algorithm>

#define MASTER 0

// Data generation function
void generate_data(double* data, int N, const std::string& input_type) {
    if (input_type == "random") {
        for (int i = 0; i < N; i++) {
            data[i] = rand() % 100000001; // Generate numbers between 0 and 100,000,000
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

// Print array function
void print_array(const double* arr, int N, const char* description, int limit = 10) {
    printf("%s: [", description);
    for (int i = 0; i < N && i < limit; i++) {
        printf("%.2f", arr[i]);
        if (i < N - 1 && i < limit - 1) {
            printf(", ");
        }
    }
    if (N > limit) {
        printf(", ..."); // Indicate more elements if the array is larger than the limit
    }
    printf("]\n");
}

// Compare and swap function
void compare_and_swap(double* arr, int i, int j, bool dir) {
    if (dir == (arr[i] > arr[j])) {
        std::swap(arr[i], arr[j]);
    }
}

// Bitonic merge function
void bitonic_merge(double* arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compare_and_swap(arr, i, i + k, dir);
        }
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

// Bitonic sort function
void bitonic_sort(double* arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonic_sort(arr, low, k, true);
        bitonic_sort(arr, low + k, k, false);
        bitonic_merge(arr, low, cnt, dir);
    }
}

int main(int argc, char *argv[]) {
    // Start Caliper function profiling
    CALI_CXX_MARK_FUNCTION;

    MPI_Init(&argc, &argv);

    int numtasks, taskid, numworkers;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    numworkers = numtasks - 1;

    // Initialize Caliper
    cali::ConfigManager mgr;
    mgr.start();

    // Mark the start of the main computation
    CALI_MARK_BEGIN("main");

    int sizeOfArray;
    std::string input_type;
    if (argc == 3) {
        sizeOfArray = atoi(argv[1]);
        input_type = argv[2];
        if (taskid == MASTER)
            printf("Running with input size: %d and input type: %s\n", sizeOfArray, input_type.c_str());
    } else {
        printf("Please provide the size of the input array and input type\n");
        return 0;
    }

    // Allocate memory for the arrays
    double *array = (double *)malloc(sizeOfArray * sizeof(double));
    double *subarray = (double *)malloc((sizeOfArray / numtasks) * sizeof(double));

    if (taskid == MASTER) {
        // Data initialization (runtime)
        CALI_MARK_BEGIN("data_init_runtime");
        generate_data(array, sizeOfArray, input_type);
        printf("Data initialized on master with type: %s.\n", input_type.c_str());
        print_array(array, sizeOfArray, "Input array", 10);
        CALI_MARK_END("data_init_runtime");
    }

    // Communication section
    CALI_MARK_BEGIN("comm");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(array, sizeOfArray / numtasks, MPI_DOUBLE, subarray, sizeOfArray / numtasks, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    printf("Task %d received data for sorting.\n", taskid);
    CALI_MARK_END("comm");

    // Sorting the local portion using bitonic sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bitonic_sort(subarray, 0, sizeOfArray / numtasks, true);
    printf("Task %d sorted its portion of data.\n", taskid);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Gathering the sorted subarrays back to the master
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather(subarray, sizeOfArray / numtasks, MPI_DOUBLE, array, sizeOfArray / numtasks, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (taskid == MASTER) {
        // Perform the final bitonic sort on the gathered data
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        bitonic_sort(array, 0, sizeOfArray, true);
        printf("Master has completed the final bitonic sort.\n");
        print_array(array, sizeOfArray, "Output array", 10);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Correctness check section
        CALI_MARK_BEGIN("correctness_check");
        for (int i = 1; i < sizeOfArray; i++) {
            if (array[i - 1] > array[i]) {
                printf("The array is not sorted!\n");
                break;
            }
        }
        printf("Correctness check completed.\n");
        CALI_MARK_END("correctness_check");
    }

    // Free allocated memory
    free(array);
    free(subarray);

    // Adiak metadata collection before stopping Caliper
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("algorithm", "bitonic_sort");
    adiak::value("programming_model", "mpi");
    adiak::value("data_type", "double");
    adiak::value("size_of_data_type", sizeof(double));
    adiak::value("input_size", sizeOfArray);
    adiak::value("input_type", input_type);
    adiak::value("num_procs", numtasks);
    adiak::value("scalability", "strong");
    adiak::value("group_num", 1);
    adiak::value("implementation_source", "handwritten");

    // Flush and stop Caliper after Adiak metadata is collected
    mgr.flush();
    mgr.stop();

    // Finalize MPI
    MPI_Finalize();

    // Mark the end of the main section
    CALI_MARK_END("main");

    if (taskid == MASTER) {
        printf("Metadata collected:\n");
        printf("Algorithm: bitonic_sort\n");
        printf("Programming model: mpi\n");
        printf("Data type: double\n");
        printf("Input size: %d\n", sizeOfArray);
        printf("Input type: %s\n", input_type.c_str());
        printf("Number of processors: %d\n", numtasks);
    }

    return 0;
}
