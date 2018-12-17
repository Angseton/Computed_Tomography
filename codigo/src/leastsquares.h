#ifndef LEAST_SQUARES
#define LEAST_SQUARES

#include <tuple>
#include <math.h>
#include <algorithm>
#include "../src/linalg.h"

using namespace std;

uint get_optimal_cut_point(vector_t singular_values){
    vector_t condition_subseq;
    for (int i = 0; i < singular_values.size(); ++i){
        condition_subseq.push_back(singular_values[0] / singular_values[i]);
    }
    point_t p1 = make_pair(1, condition_subseq[0]);
    point_t p2 = make_pair(condition_subseq.size(), condition_subseq[condition_subseq.size() - 1]);
    vector_t distance_to_segment;

    for (int i = 0; i < condition_subseq.size(); ++i){
        // Compute the distance of p = sigma_i, k(S_i) to the segment between p1, p2
        // as | (p1-p2) x (p1-p) | /  | p1 - p2|_2
        point_t p = make_pair(i + 1, condition_subseq[i]);
        double cross_product = (p1.first - p2.first) * (p1.second - p.second) - (p1.second - p2.second) * (p1.first - p.first);
        double norm2_p1subp2 = sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
        distance_to_segment.push_back(abs(cross_product) / norm2_p1subp2);
    }
    return std::max_element(
        distance_to_segment.begin(), distance_to_segment.end()) - distance_to_segment.begin();
}


vector_t least_squares(Matrix A, vector_t& b, uint max_comp, bool use_optimal_approximation){
    /*
     * Least squares solver using SVD decomposition
     */
    
    uint N = A.shape().first; // A rows
    uint M = A.shape().second; // A columns
    Matrix AtA = A.at_dot_a();
    Matrix Vt = Matrix(M, M);
    vector_t sigma;
    max_comp = max_comp > 0 ? max_comp : M;
    for (int i = 0; i < max_comp; ++i){
        vector_t v_i = generate_random_guess(M);
        std::cerr << "Computing eigenvalue " << i << " / " << M <<  "                     \r";
        pair<double, vector_t> component = AtA.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
        std::cout << "Eigenvalue " << i << " : " << component.first << endl;
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
    if (use_optimal_approximation){
        std::cerr << "Computing optimal cut-off point                           \r";
        auto ocp = get_optimal_cut_point(sigma);
        cout << "Optimal cut-off point: " << ocp << "                " <<endl;
        // Discard components
        sigma.resize(ocp);
    }
    std::cerr << "Computing Ut.." <<  "                                               \r"; 
    Matrix Ut = Matrix(N, N);
    for (int i = 0; i < sigma.size(); ++i){
        vector_t v_i = (1 / sigma[i]) * A.dot(Vt.getRow(i));
        Ut.setRow(i, v_i);
    }
    vector_t c = Ut.dot(b);
    vector_t res = vector_t(M, 0);
    for (int i = 0; i < sigma.size(); ++i){
        res = res + (c[i] / sigma[i]) * Vt.getRow(i);
    }
    // Condition number
    cout << endl << "Condition Number: " << (sigma[0] / sigma[sigma.size() - 1]) << endl;
    return res;
}

#endif