#include <cmath>
#include "rays.h"
#include "imageHandling.h"
#include "leastsquares.h"

using namespace std;

string getCmdOption(char ** begin, char ** end, const std::string & option){
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end){
        string cmd(*itr);
        return cmd;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option){
    return std::find(begin, end, option) != end;
}

int main(int argc, char * argv[]){

    string f_input;
    string f_output;
    string f_rays_csv;
    uint d_w; // Discretization Witdth
    uint d_h; // Discretization Height
    uint max_comp = 0; // Max components
    double error_sigma = 0; // Max components
    bool use_optimal_approximation = false;

    if(cmdOptionExists(argv, argv+argc, "-i")){
        f_input = getCmdOption(argv, argv + argc, "-i");
    }
    if(cmdOptionExists(argv, argv+argc, "-o")){
        f_output = getCmdOption(argv, argv + argc, "-o");
    }
    if(cmdOptionExists(argv, argv+argc, "-rays")){
        // Read rays coordinates from a csv with lines
        // of the form: 
        // x_start, y_start, x_end, y_end
        f_rays_csv = getCmdOption(argv, argv + argc, "-rays");
    }
    if(cmdOptionExists(argv, argv+argc, "-d_w")){
        d_w = (uint) stoi(getCmdOption(argv, argv + argc, "-d_w"));
    }
    if(cmdOptionExists(argv, argv+argc, "-d_h")){
        d_h = (uint) stoi(getCmdOption(argv, argv + argc, "-d_h"));
    }
    if(cmdOptionExists(argv, argv+argc, "-max_comp")){
        max_comp = (uint) stoi(getCmdOption(argv, argv + argc, "-max_comp"));
    }
    if(cmdOptionExists(argv, argv+argc, "-error_sigma")){
        error_sigma = stod(getCmdOption(argv, argv + argc, "-error_sigma"));
    }
    if(cmdOptionExists(argv, argv+argc, "-use_ocp")){
        use_optimal_approximation = true;
    }

    cout <<  "Input File: " << f_input << endl;
    cout <<  "Output File: " << f_output << endl;
    cout <<  "Discretization Size: " << d_w << "x"  << d_h << endl;
    cout <<  "Epsilon: " << error_sigma << endl;
    Matrix in = buildMatrixFromImage(f_input);
    vector<pair<point_t, point_t>> rays = read_rays_csv_file(f_rays_csv);
    uint N = d_h;
    uint M = d_w;
    Matrix phi = Matrix(rays.size(), N*M);
    vector_t b = vector_t(rays.size(), 0);
    for (int i = 0; i < rays.size(); ++i){
        std::cerr << "Simulate ray: " << i << '/' << rays.size() <<"              \r";
            pair<vector_t, double> sample = simulate_ray(
                in, N, M, rays[i].first, rays[i].second, error_sigma
            );
            phi.setRow(i, sample.first);
            b[i] = sample.second;
    }
    vector_t ls = least_squares(phi, b, max_comp, use_optimal_approximation);
    Matrix recostruction = Matrix(N, M);
    for (int i = 0; i < ls.size(); ++i){
        uint row = (int)(i / M);
        uint col = i % M;
        recostruction.set(row, col, ls[i]);
    }
    buildImageFromMatrix(recostruction, f_output);
    return 0;

}