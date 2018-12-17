
#include <gtest/gtest.h>
#include <vector>
#include "../src/leastsquares.h"
#include "../src/imageHandling.h"

/*
 * Test Linear Algebra operations/functions
 */

void EXPECT_NEAR_VECTOR(vector_t& x, vector_t& y, double epsilon){
    EXPECT_EQ(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i){
        EXPECT_NEAR(x[i], y[i], epsilon);
    }
}


TEST(linalg_matrix, test_matrix_creation) {
    Matrix m = Matrix(3, 4);
    EXPECT_EQ(m.shape().first, 3);
    EXPECT_EQ(m.shape().second, 4);
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 4; ++j){
            EXPECT_EQ(m.get(i, j), 0);
        }
    }
}

TEST(linalg_matrix, test_matrix_set_get) {
    Matrix m = Matrix(3, 4);
    m.set(0, 0, 10);
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 4; ++j){
            if (i != 0 || j != 0) EXPECT_EQ(m.get(i, j), 0);
            else EXPECT_EQ(m.get(i, j), 10);
        }
    }
}


TEST(linalg_matrix, test_matrix_sum) {
    Matrix m1 = Matrix(3, 4);
    Matrix m2 = Matrix(3, 4);
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 4; ++j){
            m1.set(i, j, 1);
            m2.set(i, j, 2);
        }
    }
    Matrix m3 = m1 + m2;
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 4; ++j){
            EXPECT_EQ(m3.get(i, j), 3);
        }
    }
}

TEST(linalg_matrix, test_matrix_scalar_product) {
    Matrix m1 = Matrix(10, 10);
    for (int i = 0; i < 10; ++i){
        for (int j = 0; j < 10; ++j){
            m1.set(i, j, 3);
        }
    }
    Matrix m3 = 2 * m1;
    for (int i = 0; i < 10; ++i){
        for (int j = 0; j < 10; ++j){
            EXPECT_EQ(m3.get(i, j), 6);
        }
    }
}

TEST(linalg_matrix, test_matrix_vector_product_identity) {
    Matrix A = Matrix(4, 4);
    vector_t v = vector_t(4, 3);
    for (int i = 0; i < 4; ++i){
        A.set(i, i, 1);
    }
    vector_t out = A.dot(v);
    EXPECT_EQ(out, v);
}


TEST(linalg_matrix, test_matrix_vector_product_ones) {
    Matrix A = Matrix(4, 4);
    vector_t v = vector_t(4, 3);
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            A.set(i, j, 1);
        }
    }
    vector_t out = A.dot(v);
    vector_t res = vector_t(4, 12);
    EXPECT_EQ(out, res);
}

TEST(linalg_matrix, test_matrix_product) {
    Matrix A = Matrix(4, 4);
    Matrix B = Matrix(4, 4);
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            A.set(i, j, 1);
            B.set(i, j, 1);
        }
    }
    Matrix out = A.dot(B);
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            EXPECT_EQ(out.get(i, j), 4);
        }
    }
}

TEST(linalg_matrix, At_dot_A){
    Matrix m(4, 3);
    vector_t r_0 = {1, 3, 6};
    vector_t r_1 = {6, 1, 5};
    vector_t r_2 = {9, 0, 2};
    vector_t r_3 = {4, 1, 1};
    m.setRow(0, r_0);
    m.setRow(1, r_1);
    m.setRow(2, r_2);
    m.setRow(3, r_3);

    Matrix res(3, 3);
    vector_t res_0 = {134, 13, 58};
    vector_t res_1 = {13, 11, 24};
    vector_t res_2 = {58, 24, 66};
    res.setRow(0, res_0);
    res.setRow(1, res_1);
    res.setRow(2, res_2);

    Matrix AtA = m.at_dot_a();

    EXPECT_TRUE(AtA == res);
}

TEST(linalg_matrix, eigen_values){
    Matrix m(3, 3);
    m.set(0, 0, 2);
    m.set(1, 1, -4);
    m.set(2, 2, 5);
    vector_t v_i = generate_random_guess(3);
    auto par = m.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
    EXPECT_NEAR(par.first, 5, 10e-4);
    m.deflate(par.second, par.first);
    auto par_2 = m.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
    EXPECT_NEAR(par_2.first, -4, 10e-4);
    m.deflate(par_2.second, par_2.first);
    auto par_3 = m.dominant_eigenvalue(v_i, ITERS_MAX, DELTA_MAX);
    EXPECT_NEAR(par_3.first, 2, 10e-4);
}

TEST(linalg_matrix, test_least_squares) {
    // y = x + 1
    Matrix A = Matrix(2, 2);
    vector_t v1 = vector_t({1, 1});
    vector_t v2 = vector_t({2, 1});
    A.setRow(0, v1);
    A.setRow(1, v2);
    vector_t b = {2, 3};
    vector_t out = least_squares(A, b, 0, false);
    vector_t res = {1, 1};
    EXPECT_NEAR_VECTOR(out, res, 10e-3);
}

TEST(linalg_matrix, test_least_squares_large_matrix) {
    // y = 2 * x + 1
    uint N = 30;
    uint M = 3;
    Matrix A = Matrix(N, M);
    vector_t b = vector_t(N, 0);
    vector_t res = {2, 4, 1};
    for (int i = 1; i < N; ++i){
        A.set(i, 0, i);
        A.set(i, 1, log(i));
        A.set(i, 2, 1);
        b[i] = 2 * i + 4 * log(i) + 1;
    }
    vector_t out = least_squares(A, b, 0, false);
    EXPECT_NEAR_VECTOR(out, res, 0.01);
}



TEST(linalg_matrix, printMatrix){
    //Test para chequear biyectividad(?) de las funciones de imageHandling
    Matrix m = buildMatrixFromImage("../../recursosTP3/data/phantom.png");
    buildImageFromMatrix(m, "../../recursosTP3/data/phantomBis.png");
    EXPECT_TRUE(true);
}