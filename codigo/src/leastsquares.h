#include <tuple>
#include <math.h>
#include <algorithm>
#include "../src/linalg.h"

using namespace std;

vector_t least_squares(Matrix A, vector_t& b){
    /*
     * Least squares solver using SVD decomposition
     */
    
    uint N = A.shape().first; // A rows
    uint M = A.shape().second; // A columns
    Matrix At = A;
    At.transpose();
    Matrix AAt = A.dot(At);
    // Use the power method to get the eigenbasisi of the AAt
    Matrix Ut = Matrix(N, N);
    vector_t sigma;
    for (int i = 0; i < N; ++i){
        std::cerr << "Computing eigenvector: " << i << '/' << N <<"              \r";
        vector_t v_i = generate_random_guess(N);
        pair<double, vector_t> component = AAt.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
        if (abs(component.first) < TOLERANCE){
            std::cerr << "Matrix AAt is not full rank.                         \r"; 
            break;
        }
        Ut.setRow(i, component.second);
        sigma.push_back(component.first);
        AAt.deflate(component.second, component.first);
    }
    for (int i = 0; i < sigma.size(); ++i){
        sigma[i] = sqrt(abs(sigma[i]));
    }

    Matrix Vt = Matrix(M, M);
    for (int i = 0; i < sigma.size(); ++i){
        vector_t v_i = (1 / (sigma[i])) * At.dot(Ut.getRow(i));
        Vt.setRow(i, v_i);
    }

    vector_t c = Ut.dot(b);
    vector_t y = vector_t(M, 0);
    vector_t res = vector_t(M, 0);
    for (int i = 0; i < sigma.size(); ++i){
        res = res + (c[i] / sigma[i]) * Vt.getRow(i);
    }
    return res;
}

