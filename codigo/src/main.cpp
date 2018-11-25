#include <cmath>
#include "rays.h"
#include "imageHandling.h"
#include "leastsquares.h"

using namespace std;

int main(int argc, char const *argv[]){
	Matrix in = buildMatrixFromImage("../../recursosTP3/data/phantom.png");
	// print_matrix(in)
	uint N = 32;
	uint M = 32;
	vector<pair<point_t, point_t>> rays;

	uint sample_resolution = 10;
	uint s = 256 / sample_resolution;
	for (int i = 0; i < s; ++i){
		uint start_y = (uint) (i * s);
		for (int j = 0; j < s; ++j){
			if (i == 1 && j == 2){
				uint end_y = (uint) (j * s);
				rays.push_back(make_pair(point_t(0, start_y), point_t(256, end_y)));
			}
		}
	}
	// cout << rays.size() << endl;
	Matrix phi = Matrix(rays.size(), N*M);
	vector_t b = vector_t(rays.size(), 0);
	for (int i = 0; i < rays.size(); ++i){
			pair<vector_t, double> sample = simulate_ray(
				in, N, M, rays[i].first, rays[i].second
			);
			phi.setRow(i, sample.first);
			// print_vector(sample.first);
			// cout << sample.second << endl;
			b[i] = sample.second;
	}
	// print_matrix(phi);
	// print_vector(b);
	vector_t ls = least_squares(phi, b);
	print_vector(ls);
	Matrix recostruction = Matrix(N, M);
	for (int i = 0; i < ls.size(); ++i){
		uint row = (int)(i / M);
		uint col = i % M;
		recostruction.set(row, col, ls[i]);
	}
	// print_matrix(recostruction);
	buildImageFromMatrix(
		recostruction, 
		"../../recursosTP3/data/phantomReconstruction.png"
	);
	return 0;

}