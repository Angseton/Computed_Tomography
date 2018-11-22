#include <tuple>
#include <algorithm>
#include "../src/linalg.h"



vector_t least_squares(Matrix A, vector_t& b){
	/*
	 * Least squares solver using SVD decomposition
	 *
	 */
	
	uint N = A.shape().first; // A rows
	uint M = A.shape().second; // A columns
	Matrix AtA = Matrix(M, M);
    // Using At won't trash the cache.
    print_matrix(A);
    A.transpose();
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < M; ++j){
            for (int k = 0; k < N; ++k){
                AtA.set(i, j,AtA.get(i, j) + A.get(i, k) * A.get(j, k));
            }
        }
    }
    print_matrix(AtA);

   	// Use the power method to get the eigenbasisi of the AtA.
    Matrix Vt = Matrix(M, M);
    vector_t sigma;
    for (int i = 0; i < M; ++i){
        std::cerr << "Computing eigenvector: " << i << '/' << M << endl;//<<"              \r";
        /*
         * Compute dominant eigenvalue and its correspondent eigenvector
         * then deflate the covariance matrix.
         */
        vector_t v_i = generate_random_guess(M);
        pair<double, vector_t> component = AtA.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
        if (abs(component.first) < TOLERANCE){
        	std::cerr << "Matrix AtA is not full rank. " << endl; // "                        \r";	
        	// for (int j = i; j < M; ++j){
        		// sigma[j] = 0;
        	// }
        	break;
        }
        sigma.push_back(component.first);
        cout << "Eigenvalue: " << component.first << endl;
        cout << "Eigenvector: " << endl;
        print_vector(component.second);
        Vt.setRow(i, component.second);
        AtA.deflate(component.second, component.first);
    }
    print_matrix(Vt);
    print_vector(sigma);
    A.transpose();
    Matrix AAt = Matrix(N, N);
    // TODO: fix indedeces
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < M; ++j){
            for (int k = 0; k < N; ++k){
                AAt.set(i, j, AAt.get(i, j) + A.get(i, k) * A.get(j, k));
            }
        }
    }
    // Use the power method to get the eigenbasisi of the AAt

    print_matrix(AAt);
    Matrix Ut = Matrix(N, N);
    for (int i = 0; i < N; ++i){
        std::cerr << "Computing eigenvector: " << i << '/' << N <<"              \r";
        vector_t v_i = generate_random_guess(N);
        pair<double, vector_t> component = AAt.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
        if (abs(component.first) < TOLERANCE){
        	std::cerr << "Matrix AAt is not full rank.                         \r";	
        	break;
        }
        Ut.setRow(i, component.second);
        AAt.deflate(component.second, component.first);
    }
    /*
     * Once we have the vectors we need from the SVD decomposition we
     * can proceed to compute the solution.
     */
    print_matrix(Ut);
    // print_matrix(Vt);

    vector_t c = Ut.dot(b);
    print_vector(b);
    print_vector(c);
    vector_t y = vector_t(M, 0);
    vector_t res = vector_t(M, 0);
    cout << endl << "Sigma: " << endl;
    print_vector(sigma);
    for (int i = 0; i < sigma.size(); ++i){
    	vector_t v = Vt.getRow(i);
    	print_vector(v);
    	cout << (c[i] / sigma[i]) << endl;
    	v = (c[i] / sigma[i]) * v;
    	print_vector(v);
    	// print_vector((c[i] / sigma[i]) * Vt.getRow(i))
    	res = res + (c[i] / sigma[i]) * Vt.getRow(i);
    }
    return res;
}


