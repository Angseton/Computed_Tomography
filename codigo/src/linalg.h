#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <tuple>

using namespace std;
// Some type definitions
typedef unsigned int uint;
typedef vector<double> vector_t;
// Random number generation
std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);
// std::uniform_real_distribution<double> distribution(1.0, 10.0);

#define TOLERANCE 10e-5
#define ITERS_MAX 1000
#define DELTA_MAX 10e-5

void print_vector(vector_t& v){
    std::cout << "---------" << std::endl;
    std::cout << "[";
    for (int i = 0; i < v.size(); ++i){
        std::cout << v[i] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "---------" << std::endl;
}


class Matrix{
public:
    Matrix(uint n, uint m); // Constructor for a n x m matrix.
    Matrix& operator = (const Matrix& other); // Assignment operator.
    Matrix operator + (const Matrix& other);
    Matrix operator * (const double scalar) const;
    Matrix dot(const Matrix& other) const;
    vector_t dot(const vector_t& v);

    void set(uint i, uint j, double value);
    void setRow(uint i, vector_t& row);
    void setRowInverse(uint i, vector_t& row);
    double get(uint i, uint j) const;
    vector_t getRow(uint i);
    
    void transpose();
    pair<uint, uint> shape();
    
    
    tuple<Matrix, vector_t, Matrix> SVD();
    void deflate(vector_t& v, double lambda);
    pair<double, vector_t> dominant_eigenvalue(vector_t guess, int iter_max, 
                                               double delta_max);
    void assertValidIndices(const uint i, const uint j) const;

private:
	uint n, m; // Matrix dimentions (n rows x m columns)
	vector<vector<double>> values;
	
};

Matrix::Matrix(uint n, uint m){
	this->n = n;
	this->m = m;
	values = vector<vector<double>>(n, vector<double>(m, 0));
}

Matrix& Matrix::operator = (const Matrix& other) {
    this->n = other.n;
    this->m = other.m;
    this->values = other.values;
    return *this;
}

void Matrix::set(uint i, uint j, double value){
    assertValidIndices(i, j);
    values[i][j] = value;
}

void Matrix::setRow(uint i, vector_t& row){
    for (int j = 0; j < row.size(); ++j){
        values[i][j] = row[j];
    }
}

void Matrix::setRowInverse(uint i, vector_t& row){
    for (int j = 0; j < row.size(); ++j){
        values[i][j] = row[row.size() - j - 1];
    }
}

double Matrix::get(uint i, uint j) const {
    assertValidIndices(i, j);
    return values[i][j];
}


Matrix Matrix::operator + (const Matrix& other) {
    Matrix res = Matrix(n, m);
    // assertValidSumShape(other);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            res.values[i][j] = values[i][j] + other.values[i][j];
        }
    }
    return res;
}

Matrix Matrix::operator * (const double scalar) const {
    Matrix res = Matrix(n, m);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            res.values[i][j] = values[i][j] * scalar;
        }
    }
    return res;
}

Matrix operator * (const double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

vector_t operator * (const double scalar, const vector_t& v) {
    vector_t res(v);
    for (int i = 0; i < res.size(); ++i){
        res[i] *= scalar;
    }
    return res;
}

vector_t operator + (const vector_t& v1, const vector_t& v2) {
    vector_t res(v1.size(), 0);
    for (int i = 0; i < res.size(); ++i){
        res[i] = v1[i] + v2[i];
    }
    return res;
}

Matrix Matrix::dot(const Matrix& other) const {
    Matrix res = Matrix(n, other.m);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < other.m; ++j){
            for (int k = 0; k < m; ++k){
                res.values[i][j] += values[i][k] * other.values[k][j];
            }
        }
    }
    return res;
}

vector_t Matrix::dot(const vector_t& v) {
    vector_t res = vector_t(n, 0);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            res[i] += values[i][j] * v[j];
        }
    }
    return res;
}


void Matrix::transpose() {
    vector<vector_t> values_tp = vector<vector_t>(m, vector_t(n, 0));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            values_tp[j][i] = values[i][j];
        }
    }
    values = values_tp;
    m = n;
    n = values.size();
}

pair<uint, uint> Matrix::shape(){
    return make_pair(n, m);
}

void Matrix::assertValidIndices(const uint i, const uint j) const {
    if (i >= n || j >= m)
        throw std::out_of_range("Indices are out of bounds");
}

double transpose_dot(vector_t& v, vector_t& w){
    double res = 0;
    for (int i = 0; i < v.size(); ++i){
        res += v[i] * w[i];
    }
    return res;
}

double norm(vector_t& v){
    double norm_2 = 0;
    for (int i = 0; i < v.size(); ++i){
        norm_2 += v[i] * v[i];
    }
    norm_2 = sqrt(norm_2);
    return norm_2;
}

double norm_inf_distance(vector_t& x, vector_t& y){
    double max_diff = 0;
    double d;
    for (int i = 0; i < x.size(); ++i){
        d = abs(x[i] - y[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

void normalize(vector_t& v){
    double norm_2 = norm(v);
    for (int i = 0; i < v.size(); ++i) v[i] /= norm_2;
}

vector_t generate_random_guess(int N){
    /*
     * Generate a random vector by sampling each
     * component from a standard normal distribution.
     */
    vector_t res = vector_t(N, 0);
    for (int i = 0; i < N; ++i) res[i] = distribution(generator);
    return res;
}

vector_t Matrix::getRow(uint i){
    return values[i];
}


pair<double, vector_t> 
Matrix::dominant_eigenvalue(vector_t guess, int iter_max, double delta_max){
    vector_t v = guess;
    vector_t old_v = v;
    double delta = std::numeric_limits<double>::infinity();
    for (int i = 0; i < iter_max && delta > delta_max; ++i){
        old_v = v;
        v = this->dot(v);
        normalize(v);
        delta = norm_inf_distance(old_v, v);
    }
    vector_t Mv = this->dot(v);
    double lambda = transpose_dot(v, Mv) / (pow(norm(v), 2));
    return make_pair(lambda, v);
}


void Matrix::deflate(vector_t& v, double lambda){
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            values[i][j] -= lambda * v[i] * v[j];
        }
    }
}

void print_matrix(Matrix& A){
    std::cout << "---------" << std::endl;
    for (int i = 0; i < A.shape().first; ++i){
        std::cout << '[';
        for (int j = 0; j < A.shape().second; ++j){
            std::cout << A.get(i, j) << ',';
        }
        std::cout << ']' << std::endl;
    }
    std::cout << "---------" << std::endl;
}
