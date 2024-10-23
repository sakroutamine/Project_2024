#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <limits.h>
#include <string>
#include <algorithm>
#include <cmath> // For pow function
#include <cstdlib> // For setenv()

#define MASTER 0

// Function to generate data based on the input type
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

// Merge function used in merge sort
void merge(double *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    double *L = (double *)malloc(n1 * sizeof(double));
    double *R = (double *)malloc(n2 * sizeof(double));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Merge sort function
void mergeSort(double *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Function to check if a portion of the array is sorted
bool check_sorted_correctly(double* local_data, int local_size, int rank, int size, MPI_Comm comm) {
    double local_last = local_data[local_size - 1];
    double* boundary_values = nullptr;

    if (rank == MASTER) {
        boundary_values = new double[size];
    }

    MPI_Gather(&local_last, 1, MPI_DOUBLE, boundary_values, 1, MPI_DOUBLE, MASTER, comm);

    bool global_sorted = true;
    if (rank == MASTER) {
        for (int i = 1; i < size; i++) {
            if (boundary_values[i - 1] > boundary_values[i]) {
                global_sorted = false;
                break;
            }
        }
        delete[] boundary_values;
    }

    MPI_Bcast(&global_sorted, 1, MPI_C_BOOL, MASTER, comm);

    return global_sorted;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI first, so it's not included in profiling

    // Now begin Caliper profiling after MPI_Init
    CALI_CXX_MARK_FUNCTION;

    int numtasks, taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    int powerOfTwo;
    std::string input_type;
    if (argc == 3) {
        powerOfTwo = atoi(argv[1]);
        input_type = argv[2];
        if (taskid == MASTER)
            printf("Running with input array size: 2^%d and input type: %s\n", powerOfTwo, input_type.c_str());
    } else {
        printf("Please provide the power of 2 for the input array size and input type\n");
        return 0;
    }

    // Set environment variable for Caliper output file name
    std::string cali_filename = "p" + std::to_string(numtasks) + "-a" + std::to_string(argc) + "-" + input_type + ".cali";
    setenv("CALI_LOGFILE_FILENAME", cali_filename.c_str(), 1);

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN("main");

    int sizeOfArray = pow(2, powerOfTwo);
    double *array = (double *)malloc(sizeOfArray * sizeof(double));
    double *subarray = (double *)malloc((sizeOfArray / numtasks) * sizeof(double));

    // Data initialization section (runtime or IO depending on how data is loaded)
    CALI_MARK_BEGIN("data_init_runtime");
    if (taskid == MASTER) {
        generate_data(array, sizeOfArray, input_type);
        printf("Data initialized on master with type: %s.\n", input_type.c_str());
    }
    CALI_MARK_END("data_init_runtime");

    // Communication section
    CALI_MARK_BEGIN("comm");

    // Small communication: MPI_Barrier
    CALI_MARK_BEGIN("comm_small");
    MPI_Barrier(MPI_COMM_WORLD); // Example of a small communication operation
    CALI_MARK_END("comm_small");

    // Large communication: MPI_Scatter and MPI_Gather
    CALI_MARK_BEGIN("comm_large");
    MPI_Scatter(array, sizeOfArray / numtasks, MPI_DOUBLE, subarray, sizeOfArray / numtasks, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    printf("Task %d received data for sorting.\n", taskid);
    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");

    // Computation section
    CALI_MARK_BEGIN("comp");

    // Small computation: checking if a small part of the array is sorted
    CALI_MARK_BEGIN("comp_small");
    bool is_sorted = true;
    int check_size = std::min(10, sizeOfArray / numtasks);
    for (int i = 1; i < check_size; i++) {
        if (subarray[i - 1] > subarray[i]) {
            is_sorted = false;
            break;
        }
    }
    printf("Task %d checked small portion of its data, sorted: %s\n", taskid, is_sorted ? "true" : "false");
    CALI_MARK_END("comp_small");

    // Large computation: performing the full merge sort on the local array
    CALI_MARK_BEGIN("comp_large");
    mergeSort(subarray, 0, (sizeOfArray / numtasks) - 1);
    printf("Task %d sorted its portion of data.\n", taskid);
    CALI_MARK_END("comp_large");

    CALI_MARK_END("comp");

    // Correctness check
    CALI_MARK_BEGIN("correctness_check");
    bool sorted_correctly = check_sorted_correctly(subarray, sizeOfArray / numtasks, taskid, numtasks, MPI_COMM_WORLD);
    if (taskid == MASTER) {
        if (sorted_correctly) {
            printf("Array is sorted correctly across all tasks.\n");
        } else {
            printf("Array is NOT sorted correctly across all tasks.\n");
        }
    }
    CALI_MARK_END("correctness_check");

    // Final communication to gather sorted arrays
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather(subarray, sizeOfArray / numtasks, MPI_DOUBLE, array, sizeOfArray / numtasks, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (taskid == MASTER) {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        mergeSort(array, 0, sizeOfArray - 1); // Final merge by master
        printf("Master has completed the final merge.\n");
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");
    }

    free(array);
    free(subarray);

    // Adiak metadata collection
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("algorithm", "merge_sort");
    adiak::value("programming_model", "mpi");
    adiak::value("data_type", "double");
    adiak::value("size_of_data_type", sizeof(double));
    adiak::value("input_size", sizeOfArray);
    adiak::value("input_type", input_type);
    adiak::value("num_procs", numtasks);
    adiak::value("scalability", "strong"); // or "weak" based on your test
    adiak::value("group_num", 5); // replace with your group number
    adiak::value("implementation_source", "handwritten");

    mgr.stop();
    mgr.flush();

    MPI_Finalize();  // Finalize MPI after profiling ends

    return 0;
}
