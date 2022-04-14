// 19120336_Lab03.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

//------------------------------------ Support function -----------------------------------//
#define PI 3.14159265359;

int k = sqrt(2);

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

Mat addGaussianBlur(Mat source, float scale, int kernelSize)
{
    Mat blurImage, preBlur;
    source.convertTo(preBlur, CV_32FC1);
    GaussianBlur(preBlur, blurImage, Size(kernelSize, kernelSize), scale, 0, BORDER_DEFAULT);

    preBlur.release();
    return blurImage;
}

Mat smoothImage(Mat image, float scale)
{
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    Mat blurImage;
    blurImage = addGaussianBlur(grayImage, scale, 3);

    grayImage.release();
    return blurImage;
}


bool isLocalMax(Mat R, int y, int x, float* value) {
    if (value == NULL)
        value = new float(R.at<float>(y, x));

    
    for (int r = -3 / 2; r <= 3 / 2; ++r) {
        for (int c = -3 / 2; c <= 3 / 2; ++c)
        {
            if (y + r < 0 || y + r >= R.rows || x + c < 0 || x + c >= R.cols)
                continue;
            if (R.at<float>(y + r, x + c) > *value)
                return false;
        }
    }
    return true;
}

void showImage(Mat image,const char* display) {
    image.convertTo(image, CV_8UC3);
    namedWindow(display, WINDOW_AUTOSIZE);
    imshow(display, image);
    waitKey(0);
}

//------------------------------------ Harris function ------------------------------------//

void mergeWithHarris(Mat origin, Mat des) {

    Mat present = origin.clone();

    for (int i = 0; i < des.rows; ++i)
        for (int j = 0; j < des.cols; ++j)
        {
            if (des.at<float>(i, j) == 1)
                circle(present, Point(j, i), 3, Scalar(0, 255, 0), 2);
        }

    namedWindow("Display Harris detection result", WINDOW_AUTOSIZE);
    imshow("Display Harris detection result", present);

    present.release();
    waitKey(0);
}

Mat detectHarris(Mat source, double thresholdRatio, float k) {

    int height = source.rows;
    int width = source.cols;
    Mat result = Mat::zeros(source.size(), CV_32FC1);

    // 1. Convert RGB image to grayscale and then apply gaussian blur to reduce noise
    Mat smoothenSource = smoothImage(source, 1.0);

    // 2. calculate Ix, Iy of source image
    Mat Ix, Iy, Ix2, Iy2, Ixy;

    // Calculate Ix, Iy using sobel
    Sobel(smoothenSource, Ix, CV_32FC1, 1, 0);
    Sobel(smoothenSource, Iy, CV_32FC1, 0, 1);


    // Calcualte Ix^2 in Ix2, Iy^2 in Iy2, Ix * Iy in Ixy with cv::multiply() function
    multiply(Ix, Ix, Ix2);
    multiply(Iy, Iy, Iy2);
    multiply(Ix, Iy, Ixy);

    GaussianBlur(Ix2, Ix2, Size(3, 3), 1);
    GaussianBlur(Iy2, Iy2, Size(3, 3), 1);
    GaussianBlur(Ixy, Ixy, Size(3, 3), 1);

    /* 3. Calculate R : R(x, y) = det(M) - k.(trace(M)) ^ 2
    * With det(M) = Ix * Iy - Ixy ^ 2
    * With trace(M) = Ix + Iy
    */

    Mat R = Mat::zeros(height, width, CV_32FC1);

    double maxR = -INT_MAX;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double detM = 1.000 * (double)Ix2.at<float>(i, j) * (double)Iy2.at<float>(i, j) - 1.000 * (double)Ixy.at<float>(i, j) * (double)Ixy.at<float>(i, j);

            double traceM = (double)Ix2.at<float>(i, j) + (double)Iy2.at<float>(i, j);

            double curRvalue = detM - k * pow(traceM, 2);

            maxR = max(maxR, curRvalue);

            R.at<float>(i, j) = (float)curRvalue;
        }
    }

    //4. Thresholding R matrix
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
        {
            if (R.at<float>(i, j) > float(thresholdRatio * maxR) && isLocalMax(R, i, j, NULL))
            {
                result.at<float>(i, j) = 1;
            }
        }

    return result;
}

//------------------------------------ Blob function --------------------------------------//

struct Blob {
    int x, y, radius;
};

Blob createNewBlob(int x, int y, float radius) {
    Blob a;
    a.x = x;
    a.y = y;
    a.radius = radius;
    return a;
}

Mat createLogKernel(int kernelSize, float kernelSigma) {
    float pi = PI;
    Mat logKernel = Mat::zeros(kernelSize, kernelSize, CV_32FC1);
    float sum = 0.0, var = 2 * kernelSigma * kernelSigma;
    for (int i = -kernelSize / 2; i <= kernelSize / 2; ++i)
    {
        for (int j = -kernelSize / 2; j <= kernelSize / 2; ++j)
        {
            float sigma = (i * i + j * j) * 1.0 / (2 * kernelSigma * kernelSigma);
            float value = -1 * (1.0 / (pi * pow(kernelSigma, 4)) * (1 - sigma) * exp(-sigma));
            logKernel.at<float>(i + kernelSize / 2, j + kernelSize / 2) = value;
            sum += logKernel.at<float>(i + kernelSize / 2, j + kernelSize / 2);
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j)
        {
            logKernel.at<float>(i, j) /= sum;
        }
    }
    
    
    return logKernel;
}

Mat calculateLoG(Mat source, float sigma) {
    int kernelSize = 2 * ceil(3 * sigma) + 1;
    Mat logKernel = createLogKernel(kernelSize, sigma);
    Mat logImage;

    logImage.convertTo(logImage, CV_32FC1);

    filter2D(source, logImage, -1, logKernel);

    return logImage;
}

float getMaxPixel(Mat source) {
    float maxValue = -INT_MAX;
    int height = source.rows;
    int width = source.cols;

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            maxValue = max(maxValue, source.at<float>(i, j));

    return maxValue;
}

vector<Mat> getScaleLaplacian(Mat source, vector<float>& maxLogValues, float sigma, int nLayers)
{
    vector<Mat> logImages;

    for (int i = 1; i <= nLayers; ++i)
    {
        float scaleSigma = sigma * pow(k, i);
        Mat logImage = calculateLoG(source, scaleSigma);

        // square log image
        multiply(logImage, logImage, logImage);

        maxLogValues.push_back(getMaxPixel(logImage));
        logImages.push_back(logImage);

    }
    return logImages;
}

bool isMaxAmongNeighbors(Mat source, int y, int x, vector<Mat> neighbors)
{
    int height = source.rows;
    int width = source.cols;
    for (int i = 0; i < neighbors.size(); ++i)
        if (!isLocalMax(neighbors[i], y, x, &source.at<float>(y, x)))
            return 0;

    return isLocalMax(source, y, x, NULL);
}

vector<Blob> getLocalMaximumPoints(vector<Mat> logImages, vector<float> maxLogValues, float threshold, float sigma)
{
    int height = logImages[0].rows;
    int width = logImages[0].cols;
    int n = logImages.size();

    vector<Blob> blobs;

    for (int i = 0; i < n; ++i)
    {
        vector<Mat> neighbors;
        if (i == 0)
            neighbors.push_back(logImages[i + 1]);
        else if (i == n - 1)
            neighbors.push_back(logImages[i - 1]);
        else {
            neighbors.push_back(logImages[i + 1]);
            neighbors.push_back(logImages[i - 1]);
        }

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if (isMaxAmongNeighbors(logImages[i], y, x, neighbors) && logImages[i].at<float>(y, x) > threshold * maxLogValues[i]) 
                {
                    float radius = pow(k, i + 1) * k * sigma;
                    Blob round = createNewBlob(x, y, (int)radius);
                    blobs.push_back(round);
                }
    }

    return blobs;
}

Mat detectBlob(Mat source) {
    // 1. Convert RGB image to grayscale and then apply gaussian blur to reduce noise
    Mat smoothenSource = smoothImage(source, 1.2);

    // 2. Generate scale-normalized LoG kernels and use them to filter image
    vector<float> maxLogValues;
    vector<Mat> logImages = getScaleLaplacian(smoothenSource, maxLogValues, 1.0, 10);

    // 3. Find local maximum points
    vector<Blob> blobs = getLocalMaximumPoints(logImages, maxLogValues, 0.05, 1.0);

    // 4. Convert back to Mat for visualization
    Mat result = source.clone();
    result.convertTo(result, CV_32FC1);
    cout << "Blob size = " << blobs.size() << endl;
    for (int i = 0; i < blobs.size(); ++i)
    {
        cout << "Blob[" << i << "](x,y,R) = {" << blobs[i].x << ", " << blobs[i].y << ", " << blobs[i].radius << "}\n";
        circle(result, Point(blobs[i].x, blobs[i].y), blobs[i].radius, Scalar(0, 255, 0));
    }

    return result;
}


//------------------------------------ DOG function ---------------------------------------//

Mat calculateGaussian(Mat source, float sigma) {
    int kernelSize = 2 * ceil(3 * sigma) + 1;

    Mat result;
    source.convertTo(result, CV_32FC1);
    GaussianBlur(result, result, Size(kernelSize, kernelSize), 1.0);
    return result;
}

vector<Mat> getScaleLaplacianDoG(Mat source, vector<float>& maxLogValues, float sigma, int nLayers)
{
    vector<Mat> dogImages;

    Mat preGauss = calculateGaussian(source, sigma);

    for (int i = 1; i <= nLayers; ++i)
    {
        float scaleSigma = sigma * pow(k, i);
        Mat gauss = calculateGaussian(source, scaleSigma);

        // square DoG image
        Mat DoGImage, squareImage;
        subtract(gauss, preGauss, DoGImage);
        multiply(DoGImage, DoGImage, squareImage);

        maxLogValues.push_back(getMaxPixel(squareImage));
        
        dogImages.push_back(squareImage);
        preGauss = gauss;
    }

    return dogImages;
}

Mat detectDoG(Mat source) {
    // 1. Convert RGB image to grayscale and then apply gaussian blur to reduce noise
    Mat smoothenSource = smoothImage(source, 1.0);

    // 2. Generate scale-normalized LoG kernels and use them to filter image
    vector<float> maxLogValues;
    vector<Mat> dogImages = getScaleLaplacianDoG(smoothenSource, maxLogValues, 1.0, 8);


    // 3. Find local maximum points
    vector<Blob> blobs = getLocalMaximumPoints(dogImages, maxLogValues, 0.04, 1.0);

    // 4. Convert back to Mat
    Mat result = source.clone();
    for (int i = 0; i < blobs.size(); ++i)
        circle(result, Point(blobs[i].x, blobs[i].y), blobs[i].radius, Scalar(0, 255, 0));
    return result;
}

//------------------------------------ SIFT function --------------------------------------//

//
//double matchBySIFT(Mat img1, Mat img2, int detector) {
//
//}



int main()
{
    Mat img;
    Mat result;

    for (int i = 1; i < 2; ++i) {
        string file = "0";
        if (i == 10)
            file = "10";
        else
            file = file + to_string(i);
        string path = "TestImages/" + file + ".jpg";
        img = imread(path, IMREAD_COLOR);
        if (!img.data) {
            cout << "Khong the mo duoc hinh\n";
            return -1;
        }

        //result = detectBlob(img);
        //showImage(result, "Display Blob Detection image");

        //result = detectHarris(img, 0.01, 0.04);
        //mergeWithHarris(img, result);

        result = detectDoG(img);
        showImage(result, "Display DoG Blob Detection image");
    }

    img.release();
    result.release();
    return 1;
}
