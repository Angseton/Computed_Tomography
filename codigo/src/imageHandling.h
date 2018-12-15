#ifndef TP3_IMAGEHANDLING_H
#define TP3_IMAGEHANDLING_H

#include "linalg.h"
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

typedef unsigned int uint;

Matrix buildMatrixFromImage(string imageFile){
    Mat image = imread(imageFile, 0); //El flag en 0 indica grayscale
    if(image.data == nullptr){
        throw runtime_error("Could not load the image.");
    }
    auto rows = (uint)image.rows;
    auto cols = (uint)image.cols;
    Matrix matrix = Matrix(rows, cols);
    for(uint i = 0; i < rows; ++i){
        for(uint j = 0; j < cols; ++j){
            matrix.set(i, j, (double)image.at<uchar>(i, j));
        }
    }
    return matrix;
}

void buildImageFromMatrix(const Matrix& m, string outputFileName){
    uint rows = m.shape().first;
    uint cols = m.shape().second;
    Mat image(rows, cols, CV_8UC1);
    for(uint i = 0; i < rows; ++i){
        for(uint j = 0; j < cols; ++j){
        	double currentPixel = m.get(i, j);
        	if(currentPixel < 0){
        		image.at<uchar>(i, j) = 0;
        	} else if (currentPixel > 255){
        		image.at<uchar>(i, j) = 255;
        	} else {
            	image.at<uchar>(i, j) = (uchar)m.get(i, j);	
        	}
        }
    }
    imwrite(outputFileName, image);
}

#endif //TP3_IMAGEHANDLING_H
