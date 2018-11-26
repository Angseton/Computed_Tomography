#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <stdexcept>

#include <cmath>
#include <algorithm>
#include "linalg.h"

using namespace std;

typedef pair<double, double> point_t;

double fix(double x){
	if (x < 0) return ceil(x);
	else return floor(x);
}

double eval_line(pair<double, double> l, double x){
	return l.first * x + l.second;
}

pair<double, double> line_fit(point_t start, point_t end){
	double m;
	if (abs((double)end.second - (double)start.second) < TOLERANCE){
		m = 0;
	}
	else{
		m = ((double)end.second - (double)start.second) /  ((double)end.first - (double)start.first);	
	}
	double b = end.second - m * end.first;
	return make_pair(m, b);

}

double time_taken(double val){
	return val + 1;
}

pair<vector_t, double> simulate_ray(Matrix& Image, uint n, uint m, point_t start, point_t end){
	uint rows = Image.shape().first;
	uint cols = Image.shape().second;
	double delta_rows = rows / n;
	double delta_cols = cols / n;

	if (start.first == end.first){
		end.first += 1;
	}
	if (start.second == end.second){
		end.second += 1;
	}

	pair<double, double> l = line_fit(start, end);
	pair<double, double> l_inv = make_pair(1 / l.first, - l.second / l.first);
	Matrix D = Matrix(n, m);

	// Filas en las que esta en rayo
	uint i1 = eval_line(l, 0);
	uint i2 = eval_line(l, cols);
	
	uint i_min = max(1.0, min((double) rows-1, floor(min(i1,i2))));
   	uint i_max = max(1.0, min((double) rows-1, ceil(max(i1,i2))));
   	
   	double t = 0;
   	for (auto i = i_min; i < i_max; ++i){
   		double j1 = eval_line(l_inv, i);
        double j2 = eval_line(l_inv, i + 1);

        uint j_min = max(0.0, (double) min((double) cols, (double) floor(min(j1, j2))));
        uint j_max = max(0.0, (double) min((double) cols, (double) ceil(max(j1, j2))));

        for (int j = j_min; j < j_max; ++j){
        	t += time_taken(Image.get(i,j));
            uint n_i = (uint) floor(min((double) n-1, (double) fix(i/delta_rows)));
            uint m_j = (uint) floor(min((double) m-1, (double) fix(j/delta_cols)));
            D.set(n_i, m_j, D.get(n_i, m_j) + 1);
        }
   	}
   	vector_t res(n * m);
   	for (int i = 0; i < D.shape().first; ++i){
   		for (int j = 0; j < D.shape().second; ++j){
   			res[(i * n) + j] = D.get(i, j);
   		}
   	}
   	return make_pair(res, t);
}


vector<pair<point_t, point_t>> read_rays_csv_file(const string& rays_csv_file){
	/**
     * Parser para el csv con coordenadas de los rayos. 
     * 
     **/
	vector<pair<point_t, point_t>> rays;	
    string line;
    ifstream infile;
    infile.open(rays_csv_file);
    if (infile.fail()) throw runtime_error("Ocurrió un error al abrir el archivo de coordenadas.");
    while (getline(infile,line)) {
        // Leo una línea y cargo una entrada
        double start_x(stod(string(strtok(&line[0u], ","))));
        double start_y(stod(string(strtok(NULL, ","))));
        double end_x(stod(string(strtok(NULL, ","))));
        double end_y(stod(string(strtok(NULL, ","))));
 		rays.push_back(make_pair(point_t(start_x, start_y), point_t(end_x, end_y)));
 		// cout << start_x << " " << start_y << " | " << end_x << " " << end_y << endl; 
    }
    infile.close();
    return rays;

}








