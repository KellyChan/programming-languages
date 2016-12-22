/*

Ref: https://www.cs.upc.edu/~jordicf/Teaching/programming/pdf4/IP10_Matrices-4slides.pdf
*/


// Declaration of a matrix with 3 rows and 4 columns
// vector< vector<int> > my_matrix(3, vector<int>(4));  

typedef vector<int> Row;      // One row of the matrix
typedef vector<Row> Matrix;   // Matrix: a vector of rows
Matrix my_matrix(3, Row(4));  // Matrix with 3 rows and 4 columns


// N-Dimensional Vectors
typedef vector<int> Dim1;
typedef vector<Dim1> Dim2;
typedef vector<Dim2> Dim3;
typedef vector<Dim3> Matrix4D;
Matrix4D my_matrix(5, Dim3(i+1, Dim2(n, Dim1(9))));


// Sum of matrices (by rows)
typedef vector< vector<int> > Matrix;
// Pre: a and b are non-empty matrices and have the same size
// Returns a+b (sum of matrices)
Matrix matrix_sum(const Matrix& a, const Matrix& b)
{
    int nrows = a.size();
    int ncols = a[0].size();
    Matrix c(nrows, vector<int>(ncols));

    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    return c;
}


// Sum of matrices (by columns)
typedef vector< vector<int> > Matrix;
// Pre: a and b are non-empty matrices and have the same size.
// Returns a+b (sum of matrics)
Matrix matrix_sum(const Matrix& a, const Matrix& b)
{
    int nrows = a.size();
    int ncols = a[0].size();
    Matrix c(nrows, vector<int>(ncols));

    for (int j = 0; j < ncols; ++j)
    {
        for (int i = 0; i < nrows; ++i)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}


// Transpose a matrix
void swap(int& a, int& b)
{
    int c = a;
    a = b;
    b = c;
}
// Pre: m is a square matrix
// Post: m contains the transpose of the input matrix
void Transpose(Matrix& m)
{
    int n = m.size();
    for (int i = 0; i < n-1; ++i)
    {
        for (int j = i+1; j < n; ++j)
        {
            swap(m[i][j], m[j][i]);
        }
    }
}


// Matrix symmetric
// Pre: m is a square matrix
// Returns true if m is symmetric, and false otherwise
bool is_symmetric(const Matrix& m)
{
    int n = m.size();
    for (int i = 0; i < n-1; ++i)
    {
        for (int j = i+1; j < n; ++j)
        {
            if (m[i][j] != m[j][i])
            {
                return false;
            }
        }
    }
    return true;
}


// Search in a matrix
// Pre: m is a non-empty matrix
// Post: i and j define the location of a cell that contains the value x in m
//       In case x is not in m, then i = j = -1
void search(const Matrix& m, int x, int& i, int& j)
{
    int nrows = m.size();
    int ncols = m[0].size();
    for (i = 0; i < nrows; ++i)
    {
        for (j = 0; j < ncols; ++j)
        {
            if (m[i][j] == x) return;
        }
    }

    i = -1;
    j = -1;
}


// Search in a sorted matrix
// Pre: m is non-empty and sorted by rows and columns in ascending order.
// Post: i and j define the location of a cell that contains th value x in m.
//       In case x is not in m, then i = j = -1.
void search(const Matrix& m, int x, int& i, int& j)
{
    int nrows = m.size();
    int ncols = m[0].size();

    i = nrows - 1;
    j = 0;

    // Invariant: x can only be found in M[0..i, j..ncol-1]
    while (i >= 0 and j < ncols)
    {
        if (m[i][j] < x) j = j + 1;
        else if (m[i][j] > x) i = i - 1;
        else return;
    }

    i = -1;
    j = -1;
}


// Matrix multiplication
// Pre: a is non-empty n*m matrix
//      b is non-empty m*p matrix
// Returns axb (an nxp matrix)
Matrix multiply(const Matrix& a, const Matrix& b)
{
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();
    Matrix c(n, vector<int>(p));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            int sum = 0;
            for (int k = 0; k < m; ++k)
            {
                sum = sum + a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}


Matrix multiply(const Matrix& a, const Matrix& b)
{
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();
    Matrix c(n, vector<int>(p, 0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}


Matrix multiply(const Matrix& a, const Matrix& b)
{
    int n = a.size();
    int m = a[0].size();
    int p = b[0].size();
    Matrix c(n, vector<int>(p, 0));

    for (int j = 0; j < p; ++j)
    {
        for (int k = 0; k < m; ++k)
        {
            for (int i = 0; i < n; ++i)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return ;
}
