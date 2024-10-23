#include <caliper/cali.h>
#include <adiak.hpp>
#include <mpi.h>
#include <vector>
#include <algorithm>

// Function prototypes
void bitonicSort(std::vector<int>& arr, int low, int cnt, bool dir);
void bitonicMerge(std::vector<int>& arr, int low, int cnt, bool dir);
std::vector<int> divideArray(const std::vector<int>& arr, int size, int rank);
std::vector<int> parallelBitonicSort(std::vector<int>& arr);

int main(int argc, char* argv[]) {
    CALI_MARK_BEGIN("main");

    // Initialize Adiak
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("algorithm", "bitonic");
    adiak::value("programming_model", "mpi");
    adiak::value("data_type", "int");
    adiak::value("size_of_data_type", sizeof(int));
    adiak::value("input_size", 1000); // Example input size
    adiak::value("input_type", "Random");
    adiak::value("num_procs", 4); // Example number of processors
    adiak::value("scalability", "strong");
    adiak::value("group_num", 1);
    adiak::value("implementation_source", "handwritten");

    // Your main function code
    std::vector<int> arr = { /* Your input data */ 5, 2, 9, 1, 5, 6, 7, 3, 8, 4 };
    std::vector<int> sorted_arr = parallelBitonicSort(arr);

    CALI_MARK_END("main");
    return 0;
}

void bitonicSort(std::vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(arr, low, k, true); // Sort in ascending order
        bitonicSort(arr, low + k, k, false); // Sort in descending order
        bitonicMerge(arr, low, cnt, dir);
    }
}

void bitonicMerge(std::vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if (dir == (arr[i] > arr[i + k])) {
                std::swap(arr[i], arr[i + k]);
            }
        }
        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

std::vector<int> divideArray(const std::vector<int>& arr, int size, int rank) {
    int n = arr.size();
    int local_n = n / size;
    std::vector<int> local_arr(local_n);
    for (int i = 0; i < local_n; i++) {
        local_arr[i] = arr[rank * local_n + i];
    }
    return local_arr;
}

std::vector<int> parallelBitonicSort(std::vector<int>& arr) {
    // Initialize MPI
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the array into sub-sequences
    std::vector<int> local_arr = divideArray(arr, size, rank);

    // Perform bitonic sort on local sub-sequence
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    bitonicSort(local_arr, 0, local_arr.size(), true);
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Gather sorted sub-sequences
    std::vector<int> sorted_arr(arr.size());
    MPI_Gather(local_arr.data(), local_arr.size(), MPI_INT, sorted_arr.data(), local_arr.size(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge sorted sub-sequences
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        bitonicMerge(sorted_arr, 0, sorted_arr.size(), true);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");
    }

    // Finalize MPI
    MPI_Finalize();
    return rank == 0 ? sorted_arr : std::vector<int>();
}
