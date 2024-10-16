#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time(0)
#include <utility> // For std::pair

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

// Function to print the matrix
void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}

// Sort the columns of a local submatrix
void sortColumns(vector<vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int col = 0; col < cols; col++) {
        vector<int> column;
        for (int row = 0; row < rows; row++) {
            column.push_back(matrix[row][col]);
        }
        sort(column.begin(), column.end());
        for (int row = 0; row < rows; row++) {
            matrix[row][col] = column[row];
        }
    }
}

// Transpose the matrix (swaps rows and columns)
vector<vector<int>> transposeMatrix(const vector<vector<int>>& matrix) {

    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<int>> transposed(cols, vector<int>(rows));

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            transposed[col][row] = matrix[row][col];
        }
    }
    return transposed;
}
// Correctly reshape a flat matrix into new shape, filling row-wise
vector<vector<int>> reshapeMatrix(const vector<int>& flatMatrix, int r, int s) {
    vector<vector<int>> reshaped(r, vector<int>(s));
    int index = 0;

    for (int row = 0; row < r; row++) {  // Fill row-wise
        for (int col = 0; col < s; col++) {
            reshaped[row][col] = flatMatrix[index++];
        }
    }
    return reshaped;
}

// Function to flatten the matrix in column-major order
vector<int> flattenMatrixColumnMajor(const vector<vector<int>>& matrix) {
    vector<int> columnMajor;
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            columnMajor.push_back(matrix[row][col]);
        }
    }
    return columnMajor;
}

// Function to flatten the matrix in row-major order
vector<int> flattenMatrixRowMajor(const vector<vector<int>>& matrix) {
    vector<int> rowMajor;
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            rowMajor.push_back(matrix[row][col]);
        }
    }
    return rowMajor;
}

// Function to reconstruct the matrix from column-major order data
void reconstructMatrixFromColumnMajor(vector<vector<int>>& matrix, const vector<int>& columnMajor) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int index = 0;

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            matrix[row][col] = columnMajor[index++];
        }
    }
}

// Step 6: Add infinities
void addInfinities(vector<vector<int>>& matrix, int r) {
    int cols = matrix[0].size();
    int halfR = r / 2;

    for (int i = 0; i < halfR; i++) {
        for (int col = 0; col < cols; col++) {
            matrix[i].insert(matrix[i].begin(), numeric_limits<int>::min());
        }
    }

    for (int i = matrix.size() - halfR; i < matrix.size(); i++) {
        for (int col = 0; col < cols; col++) {
            matrix[i].push_back(numeric_limits<int>::max());
        }
    }
}

// Function to calculate valid dimensions (r, s) based on N
pair<int, int> calculateDimensions(size_t N) {
    int s = static_cast<int>(std::sqrt(N));  // Start with the square root of N as an initial guess for s
    if (s < 2) s = 2; // Ensure s >= 2
    int r;

    // Adjust `s` and `r` to meet the conditions
    while (s >= 2) {
        if (N % s == 0) {
            r = N / s;
            if (r % s == 0 && r >= 2 * (s - 1) * (s - 1)) {
                // Found valid r and s
                return {r, s};
            }
        }
        s--;
    }

    // No valid (r, s) pair found
    return {-1, -1}; // Indicate failure to find valid dimensions
}

// Function to find valid reshape dimensions for reshaping steps
pair<int, int> findReshapeDimensions(int N) {
    int s = 2;
    int r;
    while (s <= N) {
        if (N % s == 0) {
            r = N / s;
            if (r % s == 0 && r >= 2 * (s - 1) * (s - 1)) {
                return {r, s};
            }
        }
        s++;
    }
    // If no valid dimensions found, return -1
    return {-1, -1};
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    // Initialize Adiak
    adiak::init(NULL);

    // Collect Adiak Metadata
    string algorithm = "column_sort";
    string programming_model = "mpi";
    string data_type = "int";
    int size_of_data_type = sizeof(int);
    string input_type = "Random"; // Since data is randomly generated
    int scalability = 1; // 1 for strong scaling, 2 for weak scaling
    int group_number = 5;
    string implementation_source = "handwritten";

    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("algorithm", algorithm); // The name of the algorithm you are using
    adiak::value("programming_model", programming_model); // e.g. "mpi"
    adiak::value("data_type", data_type); // The datatype of input elements
    adiak::value("size_of_data_type", size_of_data_type); // sizeof(datatype) of input elements in bytes
    adiak::value("input_size", static_cast<int>(argc > 1 ? atoi(argv[1]) : 0)); // The number of elements in input dataset
    adiak::value("input_type", input_type); // Input type
    adiak::value("num_procs", 0); // To be updated after MPI_Init
    adiak::value("scalability", scalability); // "strong" or "weak"
    adiak::value("group_num", group_number); // The number of your group
    adiak::value("implementation_source", implementation_source); // "online", "ai", "handwritten"

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cali::ConfigManager mgr;
    mgr.start();

    // Update Adiak metadata with the actual number of processes
    adiak::value("num_procs", size);

    // Check for correct number of arguments
    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " N" << endl;
            cerr << "Please provide the input size N as a command-line argument." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]); // Convert input string to integer

    if (N <= 0) {
        if (rank == 0) {
            cerr << "N must be a positive integer." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Find valid (r, s) pairs
    pair<int, int> validPairs = calculateDimensions(N);

    int r = validPairs.first;
    int s = validPairs.second;

    if (r == -1 || s == -1) {
        if (rank == 0) {
            cerr << "No valid (r, s) pairs found for N = " << N << endl;
            cerr << "Ensure that N can be expressed as r × s, where s >= 2, s divides r, and r >= 2*(s-1)^2." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        cout << "Using r = " << r << ", s = " << s << " (N = " << N << ")" << endl;
    }

    // Begin data initialization region
    CALI_MARK_BEGIN("data_init_runtime");

    // Generate random matrix
    vector<vector<int>> matrix(r, vector<int>(s));
    srand(time(0) + rank); // Seed random number generator

    // Fill the matrix with random integers
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < s; j++) {
            matrix[i][j] = rand() % 1000; // Random integers between 0 and 999
        }
    }

    CALI_MARK_END("data_init_runtime");

    int rows = r;
    int cols = s;

    // Validate that the number of processes evenly divides the number of columns
    if (cols % size != 0) {
        if (rank == 0) {
            cerr << "Number of columns (" << cols << ") is not evenly divisible by the number of MPI processes (" << size << ")." << endl;
            cerr << "Please choose a number of MPI processes that evenly divides the number of columns." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int local_cols = cols / size; // Distribute columns evenly across processes

    // Flatten the matrix in column-major order
    vector<int> column_major_data = flattenMatrixColumnMajor(matrix);

    // Allocate space for local submatrix (each process gets a set of columns)
    vector<int> local_data(local_cols * rows);

    // Begin communication region
    CALI_MARK_BEGIN("comm");

    // Begin comm_large region for Scatter
    CALI_MARK_BEGIN("comm_large");
    MPI_Scatter(column_major_data.data(), local_cols * rows, MPI_INT,
                local_data.data(), local_cols * rows, MPI_INT,
                0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");

    // Convert local_data to a local matrix for sorting
    vector<vector<int>> local_matrix(rows, vector<int>(local_cols));
    for (int col = 0; col < local_cols; col++) {
        for (int row = 0; row < rows; row++) {
            local_matrix[row][col] = local_data[col * rows + row];
        }
    }

    // Begin computation region
    CALI_MARK_BEGIN("comp");

    // Begin comp_large region for sorting
    CALI_MARK_BEGIN("comp_large");
    // Step 1: Each process sorts its local columns
    sortColumns(local_matrix);
    CALI_MARK_END("comp_large");

    CALI_MARK_END("comp");

    // Convert the local_matrix back to local_data for gathering
    for (int col = 0; col < local_cols; col++) {
        for (int row = 0; row < rows; row++) {
            local_data[col * rows + row] = local_matrix[row][col];
        }
    }

    // Begin communication region for Gather
    CALI_MARK_BEGIN("comm");

    // Begin comm_large region for Gather
    CALI_MARK_BEGIN("comm_large");
    // Gather the sorted columns back to the root process
    MPI_Gather(local_data.data(), local_cols * rows, MPI_INT,
               column_major_data.data(), local_cols * rows, MPI_INT,
               0, MPI_COMM_WORLD);
    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");

    if (rank == 0) {
        // Reconstruct the matrix from column-major order
        reconstructMatrixFromColumnMajor(matrix, column_major_data);

        cout << "Matrix after first column sort:" << endl;
        printMatrix(matrix);

        // Begin computation region for transpose
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 2: Transpose the matrix
        matrix = transposeMatrix(matrix);
        cout << "Matrix after transpose:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for reshaping
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 3: Reshape the transposed matrix
        vector<int> flatTransposed = flattenMatrixRowMajor(matrix);

        // Calculate new dimensions for reshaping
        int N_total = flatTransposed.size();

        // Find new (r', s') dimensions for reshaping
        pair<int, int> reshapeDims = calculateDimensions(N_total);
        int r_prime = reshapeDims.first;
        int s_prime = reshapeDims.second;

        if (r_prime == -1 || s_prime == -1) {
            cerr << "No valid reshape dimensions found for N = " << N_total << endl;
            MPI_Finalize();
            return 1;
        }

        // Reshape the matrix
        matrix = reshapeMatrix(flatTransposed, r_prime, s_prime);
        cout << "Matrix after reshape:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for second column sort
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 4: Sort columns again
        sortColumns(matrix);
        cout << "Matrix after second column sort:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for reshaping back
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 5: Reshape back to original dimensions and transpose back
        vector<int> flatData = flattenMatrixRowMajor(matrix);
        matrix = reshapeMatrix(flatData, s, r); // Note: Using s and r due to transpose
        cout << "Matrix after reshape back:" << endl;
        printMatrix(matrix);

        // Transpose back
        matrix = transposeMatrix(matrix);
        cout << "Matrix after transpose back:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for third column sort
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 6: Sort columns again
        sortColumns(matrix);
        cout << "Matrix after third column sort:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for adding infinities
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 7: Add infinities
        addInfinities(matrix, rows);
        cout << "Matrix after adding infinities:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin computation region for final sort
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");

        // Step 8: Final sorting after adding infinities
        sortColumns(matrix);
        cout << "Matrix after final sort:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // Begin correctness check region
        CALI_MARK_BEGIN("correctness_check");

        // Final step: Remove infinities
        for (auto& row : matrix) {
            row.erase(remove_if(row.begin(), row.end(), [](int x) {
                return x == numeric_limits<int>::min() || x == numeric_limits<int>::max();
            }), row.end());
        }

        cout << "Final sorted matrix:" << endl;
        printMatrix(matrix);

        CALI_MARK_END("correctness_check");
    }
    mgr.stop();
    mgr.flush();

    // Finalize MPI
    MPI_Finalize();

    return 0;
}