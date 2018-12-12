#ifndef RAYS
#define RAYS

#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <stdexcept>

#include <cmath>
#include <algorithm>
#include "leastsquares.h"

using namespace std;

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
	} else {
		m = ((double)end.second - (double)start.second) /  ((double)end.first - (double)start.first);	
	}
	double b = end.second - m * end.first;
	return make_pair(m, b);

}

double time_taken(double val){
	return val + 1;
}

pair<vector_t, double> simulate_ray(Matrix& Image, uint n, uint m, point_t start, point_t end, double error_sigma){
  uint rows = Image.shape().first;
  uint cols = Image.shape().second;
  double delta_rows = rows / n;
  double delta_cols = cols / m;

  if (start.first == end.first){
    end.first += 1;
  }
  if (start.second == end.second){//hace falta?
    end.second += 1;
  }

  pair<double, double> l = line_fit(start, end);
  Matrix D = Matrix(n, m);
  double t = 0;
  for (int i = 0; i < rows; ++i){
    for (int j = 0; j < cols; ++j){
      double l_left = eval_line(l, i);
      double l_right = eval_line(l, i + 1);
      if ( (j < l_left && l_left < j + 1) || (j < l_right && l_right < j + 1) ){
        t += time_taken(Image.get(i,j));
        uint n_i = (uint) floor(min((double) n-1, (double) fix(i/delta_rows)));
        uint m_j = (uint) floor(min((double) m-1, (double) fix(j/delta_cols)));
        D.set(n_i, m_j, D.get(n_i, m_j) + 1);
      }
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
    }
    infile.close();
    return rays;

}


vector<pair<point_t, point_t>> generate_random_rays(Matrix& im, uint sample_size){
	uint height = im.shape().first;
	uint width = im.shape().second;
	vector<pair<point_t, point_t>> rays;
	std::bernoulli_distribution bernoulli(0.5);
	std::uniform_int_distribution<int> choose_x(0, width);
	std::uniform_int_distribution<int> choose_y(0, height);
	for (int i = 0; i < sample_size; ++i){
		bool start_horizontal = (bool) bernoulli(generator);
		bool end_horizontal = (bool) bernoulli(generator);
        // Chose starting point coordinates
        double start_x = start_horizontal? 0 : choose_x(generator);
        double start_y = start_horizontal? choose_y(generator) : 0;
        // Chose end point coordinates
        double end_x = end_horizontal ? width : choose_x(generator);
        double end_y = end_horizontal ? choose_y(generator) : height;
        rays.push_back(make_pair(point_t(start_x, start_y), point_t(end_x, end_y)));
	}
	return rays;
}

#endif




