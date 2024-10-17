#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

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

// Correctly reshape a flat matrix into new shape, filling it column-wise
vector<vector<int>> reshapeMatrix(const vector<int>& flatMatrix, int r, int s) {
    vector<vector<int>> reshaped(r, vector<int>(s));
    int index = 0;

    for (int row = 0; row < r ; row++) {  // Fill column-wise to preserve column-major order
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

// Function to flatten the matrix in row-major order for step 2
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the matrix (6x2 matrix as an example)
    vector<vector<int>> matrix = {
        {34, 23}, 
        {7, 32}, 
        {5, 62}, 
        {78, 0}, 
        {2, 44}, 
        {19, 1}
    };

    int rows = matrix.size();
    int cols = matrix[0].size();
    int local_cols = cols / size; // Distribute columns evenly across processes

    // Flatten the matrix in column-major order
    vector<int> column_major_data = flattenMatrixColumnMajor(matrix);

    // Allocate space for local submatrix (each process gets a set of columns)
    vector<int> local_data(local_cols * rows);

    // Scatter the columns of the matrix to all processes
    MPI_Scatter(column_major_data.data(), local_cols * rows, MPI_INT,
                local_data.data(), local_cols * rows, MPI_INT,
                0, MPI_COMM_WORLD);

    // Convert local_data to a local matrix for sorting
    vector<vector<int>> local_matrix(rows, vector<int>(local_cols));
    for (int col = 0; col < local_cols; col++) {
        for (int row = 0; row < rows; row++) {
            local_matrix[row][col] = local_data[col * rows + row];
        }
    }

    // Step 1: Each process sorts its local columns
    sortColumns(local_matrix);

    // Convert the local_matrix back to local_data for gathering
    for (int col = 0; col < local_cols; col++) {
        for (int row = 0; row < rows; row++) {
            local_data[col * rows + row] = local_matrix[row][col];
        }
    }

    // Gather the sorted columns back to the root process
    MPI_Gather(local_data.data(), local_cols * rows, MPI_INT,
               column_major_data.data(), local_cols * rows, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reconstruct the matrix from column-major order
        reconstructMatrixFromColumnMajor(matrix, column_major_data);

        cout << "Matrix after first column sort:" << endl;
        printMatrix(matrix);

        // Step 2: Transpose the matrix
        matrix = transposeMatrix(matrix);
        cout << "Matrix after transpose:" << endl;
        printMatrix(matrix);
        vector<int> flatTransposed = flattenMatrixRowMajor(matrix);
        matrix = reshapeMatrix(flatTransposed, 4, 3); // Adjust the new dimensions
        cout << "Matrix after reshape:" << endl;
        printMatrix(matrix);

        // Step 3: Sort columns again
        sortColumns(matrix);
        cout << "Matrix after second column sort: " << endl;
        printMatrix(matrix);

        // Step 4: Transpose and reshape back
        vector<int> flatData = flattenMatrixRowMajor(matrix);
        matrix = reshapeMatrix(flatData, 2, 6); // Adjust to the original dimensions
        cout << "Matrix after reshape back:" << endl;
        printMatrix(matrix);

        // Now transpose back
        matrix = transposeMatrix(matrix);
        cout << "Matrix after transpose back:" << endl;
        printMatrix(matrix);

        // Step 5: Sort columns again
        sortColumns(matrix);
        cout << "Matrix after second column sort:" << endl;
        printMatrix(matrix);

        // Step 6: Add infinities
        addInfinities(matrix, rows);
        cout << "Matrix after adding infinities:" << endl;
        printMatrix(matrix);

        // Step 7: Final sorting after adding infinities
        sortColumns(matrix);
        cout << "Matrix after final sort:" << endl;
        printMatrix(matrix);

        // Final step: Remove infinities
        for (auto& row : matrix) {
            row.erase(remove_if(row.begin(), row.end(), [](int x) {
                return x == numeric_limits<int>::min() || x == numeric_limits<int>::max();
            }), row.end());
        }

        cout << "Final sorted matrix:" << endl;
        printMatrix(matrix);
    }

    MPI_Finalize();
    return 0;
}
