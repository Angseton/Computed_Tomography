#ifndef LEAST_SQUARES
#define LEAST_SQUARES

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
    //Matrix At = A.transpose();
    Matrix AtA = A.at_dot_a();
    Matrix Vt = Matrix(M, M);
    vector_t sigma;
    for (int i = 0; i < M; ++i){
        std::cerr << "Computing eigenvector: " << i << '/' << M <<"              \r";
        vector_t v_i = generate_random_guess(M);
        pair<double, vector_t> component = AtA.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
        if (component.first < TOLERANCE){
            std::cerr << "Matrix AtA is not full rank.                         \r"; 
            break;
        }
        Vt.setRow(i, component.second);
        sigma.push_back(component.first);
        AtA.deflate(component.second, component.first);
    }
    for (int i = 0; i < sigma.size(); ++i){
        sigma[i] = sqrt(sigma[i]);
    }
    
    Matrix Ut = Matrix(N, N);
    for (int i = 0; i < sigma.size(); ++i){
        vector_t v_i = (1 / sigma[i]) * A.dot(Vt.getRow(i));
        Ut.setRow(i, v_i);
    }
    //Matrix U = Ut.transpose();
    vector_t c = Ut.dot(b);
    vector_t res = vector_t(M, 0);
    for (int i = 0; i < sigma.size(); ++i){
        res = res + (c[i] / sigma[i]) * Vt.getRow(i);
    }
    // Condition number
    // cout << endl << "Condition Number: " << (sigma[0] / sigma[sigma.size() - 1]) << endl;
    return res;
}

#endif