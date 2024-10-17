#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

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

// Step 1: Sort each column of the matrix
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

// Step 2 and Step 4: Reshape and transpose
vector<vector<int>> reshapeTranspose(const vector<vector<int>>& matrix, int r, int s) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // We need to grab values column by column and flatten them
    vector<int> columnFlat;
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            columnFlat.push_back(matrix[row][col]);
        }
    }

    // Reshape into r by s submatrix
    vector<vector<int>> reshaped(r, vector<int>(s));
    int index = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < s; j++) {
            reshaped[i][j] = columnFlat[index++];
        }
    }

    return reshaped;
}

// Step 4: Reshape and transpose back
vector<vector<int>> reshapeTransposeBack(const vector<vector<int>>& matrix, int r, int s) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Flatten the matrix row by row
    vector<int> flat;
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            flat.push_back(elem);
        }
    }

    // Reshape back into r by s (original columnar structure)
    vector<vector<int>> reshaped(r, vector<int>(s));
    int index = 0;
    for (int col = 0; col < s; col++) {
        for (int row = 0; row < r; row++) {
            reshaped[row][col] = flat[index++];
        }
    }

    return reshaped;
}

// Generate input data based on the provided parameters
vector<int> generateInputData(size_t input_size, const string& input_type) {
    vector<int> data(input_size);
    
    if (input_type == "sorted") {
        // Sorted input
        iota(data.begin(), data.end(), 1);  // Fill with ascending numbers
    } 
    else if (input_type == "random") {
        // Random input
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, 1000000);
        generate(data.begin(), data.end(), [&]() { return dis(gen); });
    } 
    else if (input_type == "reverse_sorted") {
        // Reverse sorted input
        iota(data.begin(), data.end(), 1);
        reverse(data.begin(), data.end());
    } 
    return data;
}

// Automatically calculate `r` and `s` based on input size and ensure the conditions are met
pair<int, int> calculateDimensions(size_t N) {
    int s = sqrt(N);  // Start with the square root of N as an initial guess for s
    int r = N / s;

    // Adjust `s` and `r` to meet the condition: r >= 2(s - 1)^2
    while (r < 2 * pow(s - 1, 2) || N % s != 0) {
        s--;
        r = N / s;
    }

    return {r, s};
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

int main() {
    // Input size and type
    size_t input_size;
    string input_type;
    
    cout << "Enter input size (as a power of 2, e.g., 2^16): ";
    cin >> input_size;
    
    cout << "Enter input type (sorted, random, reverse_sorted, 1%perturbed): ";
    cin >> input_type;

    // Generate input data based on the size and type
    vector<int> input_data = generateInputData(input_size, input_type);

    // Automatically calculate r and s
    auto [r, s] = calculateDimensions(input_size);

    // Validate the dimensions and the condition r >= 2(s - 1)^2
    if (r < 2 * pow(s - 1, 2)) {
        cout << "Invalid dimensions: r does not satisfy r >= 2(s - 1)^2." << endl;
        return 1;
    }

    cout << "Calculated dimensions - Rows (r): " << r << ", Columns (s): " << s << endl;

    // Fill the matrix with generated data
    vector<vector<int>> matrix(r, vector<int>(s));
    size_t index = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < s; j++) {
            matrix[i][j] = input_data[index++];
        }
    }

    cout << "Initial matrix:" << endl;
    printMatrix(matrix);

    // Step 1: Sort each column
    sortColumns(matrix);
    cout << "Step 1 - Sorted columns:" << endl;
    printMatrix(matrix);

    // Step 2: Transpose and reshape
    matrix = reshapeTranspose(matrix, (r * s) / s, s);
    cout << "Step 2 - Transposed and reshaped:" << endl;
    printMatrix(matrix);

    // Step 3: Sort each column again
    sortColumns(matrix);
    cout << "Step 3 - Sorted columns after transpose:" << endl;
    printMatrix(matrix);

    // Step 4: Reshape and transpose back
    matrix = reshapeTransposeBack(matrix, r, s);
    cout << "Step 4 - Reshaped and transposed back:" << endl;
    printMatrix(matrix);

    // Step 5: Sort each column again
    sortColumns(matrix);
    cout << "Step 5 - Sorted columns again:" << endl;
    printMatrix(matrix);

    // Step 6: Add infinities
    addInfinities(matrix, r);
    cout << "Step 6 - Added infinities:" << endl;
    printMatrix(matrix);

    // Step 7: Sort each column again
    sortColumns(matrix);
    cout << "Step 7 - Sorted columns with infinities:" << endl;
    printMatrix(matrix);

    // Final step: Remove infinities and print sorted matrix
    for (auto& row : matrix) {
        row.erase(remove_if(row.begin(), row.end(), [](int x) {
            return x == numeric_limits<int>::min() || x == numeric_limits<int>::max();
        }), row.end());
    }

    cout << "Final sorted matrix:" << endl;
    printMatrix(matrix);

    return 0;
}
