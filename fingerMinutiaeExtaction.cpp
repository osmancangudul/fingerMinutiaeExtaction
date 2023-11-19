
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
namespace cv
{
    using std::vector;
}

using namespace cv;
using namespace std;

Scalar white = CV_RGB(255, 255, 255);
Scalar green = CV_RGB(0, 255, 0);

double kx = 0.8;
double ky = 0.8;
double blockSigma = 5.0;
double gradientSigma = 1.0;
double orientSmoothSigma = 5.0;
// The frequency is set to 0.11 experimentally. You can change this value
// or use a dynamic calculation instead.
double freqValue = 0.11;
int ddepth = CV_32FC1;
bool addBorder = false;
int cannyLowThreshold = 10;
int cannyRatio = 3;
int kernelSize = 3;
int blurringTimes = 30;
int dilationSize = 10;
int dilationType = 1;

float deviation(const cv::Mat &im, float average)
{
    float sdev = 0.0;

    for (int i = 0; i < im.rows; i++)
    {
        for (int j = 0; j < im.cols; j++)
        {
            float pixel = im.at<float>(i, j);
            float dev = (pixel - average) * (pixel - average);
            sdev = sdev + dev;
        }
    }

    int size = im.rows * im.cols;
    float var = sdev / (size - 1);
    float sd = std::sqrt(var);

    return sd;
}

Mat normalize_image(const Mat &im, double reqMean,
                    double reqVar)
{

    cv::Mat convertedIm;
    im.convertTo(convertedIm, CV_32FC1);

    cv::Scalar mean = cv::mean(convertedIm);
    cv::Mat normalizedImage = convertedIm - mean[0];

    cv::Scalar normMean = cv::mean(normalizedImage);
    float stdNorm = deviation(normalizedImage, normMean[0]);
    normalizedImage = normalizedImage / stdNorm;
    normalizedImage = reqMean + normalizedImage * cv::sqrt(reqVar);

    return normalizedImage;
}

void gradient(const cv::Mat &image, cv::Mat &xGradient,
              cv::Mat &yGradient)
{
    xGradient = cv::Mat::zeros(image.rows, image.cols, ddepth);
    yGradient = cv::Mat::zeros(image.rows, image.cols, ddepth);

    // Pointer access more effective than Mat.at<T>()
    for (int i = 1; i < image.rows - 1; i++)
    {
        const auto *image_i = image.ptr<float>(i);
        auto *xGradient_i = xGradient.ptr<float>(i);
        auto *yGradient_i = yGradient.ptr<float>(i);
        for (int j = 1; j < image.cols - 1; j++)
        {
            float xPixel1 = image_i[j - 1];
            float xPixel2 = image_i[j + 1];

            float yPixel1 = image.at<float>(i - 1, j);
            float yPixel2 = image.at<float>(i + 1, j);

            float xGrad;
            float yGrad;

            if (j == 0)
            {
                xPixel1 = image_i[j];
                xGrad = xPixel2 - xPixel1;
            }
            else if (j == image.cols - 1)
            {
                xPixel2 = image_i[j];
                xGrad = xPixel2 - xPixel1;
            }
            else
            {
                xGrad = 0.5f * (xPixel2 - xPixel1);
            }

            if (i == 0)
            {
                yPixel1 = image_i[j];
                yGrad = yPixel2 - yPixel1;
            }
            else if (i == image.rows - 1)
            {
                yPixel2 = image_i[j];
                yGrad = yPixel2 - yPixel1;
            }
            else
            {
                yGrad = 0.5f * (yPixel2 - yPixel1);
            }

            xGradient_i[j] = xGrad;
            yGradient_i[j] = yGrad;
        }
    }
}

cv::Mat orient_ridge(const cv::Mat &im)
{

    cv::Mat gradX, gradY;
    cv::Mat sin2theta;
    cv::Mat cos2theta;

    int kernelSize = 6 * round(gradientSigma);

    if (kernelSize % 2 == 0)
    {
        kernelSize++;
    }

    // Define Gaussian kernel
    cv::Mat gaussKernelX =
        cv::getGaussianKernel(kernelSize, gradientSigma, CV_32FC1);
    cv::Mat gaussKernelY =
        cv::getGaussianKernel(kernelSize, gradientSigma, CV_32FC1);
    cv::Mat gaussKernel = gaussKernelX * gaussKernelY.t();

    // Peform Gaussian filtering
    cv::Mat fx, fy;
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(gaussKernel, fx, -1, kernelx);
    cv::filter2D(gaussKernel, fy, -1, kernely);

    // Gradient of Gaussian
    gradient(gaussKernel, fx, fy);

    gradX.convertTo(gradX, CV_32FC1);
    gradY.convertTo(gradY, CV_32FC1);

    // Gradient of the image in x
    cv::filter2D(im, gradX, -1, fx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    // Gradient of the image in y
    cv::filter2D(im, gradY, -1, fy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat grad_xx, grad_xy, grad_yy;
    cv::multiply(gradX, gradX, grad_xx);
    cv::multiply(gradX, gradY, grad_xy);
    cv::multiply(gradY, gradY, grad_yy);

    // Now smooth the covariance data to perform a weighted summation of the data
    int sze2 = 6 * round(blockSigma);

    if (sze2 % 2 == 0)
    {
        sze2++;
    }

    cv::Mat gaussKernelX2 = cv::getGaussianKernel(sze2, blockSigma, CV_32FC1);
    cv::Mat gaussKernelY2 = cv::getGaussianKernel(sze2, blockSigma, CV_32FC1);
    cv::Mat gaussKernel2 = gaussKernelX2 * gaussKernelY2.t();

    cv::filter2D(grad_xx, grad_xx, -1, gaussKernel2, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    cv::filter2D(grad_xy, grad_xy, -1, gaussKernel2, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    cv::filter2D(grad_yy, grad_yy, -1, gaussKernel2, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);

    grad_xy *= 2;

    // Analytic solution of principal direction
    cv::Mat G1, G2, G3;
    cv::multiply(grad_xy, grad_xy, G1);
    G2 = grad_xx - grad_yy;
    cv::multiply(G2, G2, G2);

    cv::Mat denom;
    G3 = G1 + G2;
    cv::sqrt(G3, denom);

    cv::divide(grad_xy, denom, sin2theta);
    cv::divide(grad_xx - grad_yy, denom, cos2theta);

    int sze3 = 6 * round(orientSmoothSigma);

    if (sze3 % 2 == 0)
    {
        sze3 += 1;
    }

    cv::Mat gaussKernelX3 =
        cv::getGaussianKernel(sze3, orientSmoothSigma, CV_32FC1);
    cv::Mat gaussKernelY3 =
        cv::getGaussianKernel(sze3, orientSmoothSigma, CV_32FC1);
    cv::Mat gaussKernel3 = gaussKernelX3 * gaussKernelY3.t();

    cv::filter2D(cos2theta, cos2theta, -1, gaussKernel3, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    cv::filter2D(sin2theta, sin2theta, -1, gaussKernel3, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);

    sin2theta.convertTo(sin2theta, ddepth);
    cos2theta.convertTo(cos2theta, ddepth);
    cv::Mat orientim = cv::Mat::zeros(sin2theta.rows, sin2theta.cols, ddepth);

    // Pointer access more effective than Mat.at<T>()
    for (int i = 0; i < sin2theta.rows; i++)
    {
        const float *sin2theta_i = sin2theta.ptr<float>(i);
        const float *cos2theta_i = cos2theta.ptr<float>(i);
        auto *orientim_i = orientim.ptr<float>(i);
        for (int j = 0; j < sin2theta.cols; j++)
        {
            orientim_i[j] = (M_PI + std::atan2(sin2theta_i[j], cos2theta_i[j])) / 2;
        }
    }

    return orientim;
}

void meshgrid(int kernelSize, cv::Mat &meshX, cv::Mat &meshY)
{
    std::vector<int> t;

    for (int i = -kernelSize; i < kernelSize; i++)
    {
        t.push_back(i);
    }

    cv::Mat gv = cv::Mat(t);
    int total = gv.total();
    gv = gv.reshape(1, 1);

    cv::repeat(gv, total, 1, meshX);
    cv::repeat(gv.t(), 1, total, meshY);
}

cv::Mat filter_ridge(const cv::Mat &inputImage,
                     const cv::Mat &orientationImage,
                     const cv::Mat &frequency)
{

    // Fixed angle increment between filter orientations in degrees
    int angleInc = 3;

    inputImage.convertTo(inputImage, CV_32FC1);
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    orientationImage.convertTo(orientationImage, CV_32FC1);

    cv::Mat enhancedImage = cv::Mat::zeros(rows, cols, CV_32FC1);
    cv::vector<int> validr;
    cv::vector<int> validc;

    double unfreq = frequency.at<float>(1, 1);

    cv::Mat freqindex = cv::Mat::ones(100, 1, CV_32FC1);

    double sigmax = (1 / unfreq) * kx;
    double sigmax_squared = sigmax * sigmax;
    double sigmay = (1 / unfreq) * ky;
    double sigmay_squared = sigmay * sigmay;

    int szek = (int)round(3 * (std::max(sigmax, sigmay)));

    cv::Mat meshX, meshY;
    meshgrid(szek, meshX, meshY);

    cv::Mat refFilter = cv::Mat::zeros(meshX.rows, meshX.cols, CV_32FC1);

    meshX.convertTo(meshX, CV_32FC1);
    meshY.convertTo(meshY, CV_32FC1);

    double pi_by_unfreq_by_2 = 2 * M_PI * unfreq;

    for (int i = 0; i < meshX.rows; i++)
    {
        const float *meshX_i = meshX.ptr<float>(i);
        const float *meshY_i = meshY.ptr<float>(i);
        auto *reffilter_i = refFilter.ptr<float>(i);
        for (int j = 0; j < meshX.cols; j++)
        {
            float meshX_i_j = meshX_i[j];
            float meshY_i_j = meshY_i[j];
            float pixVal2 = -0.5f * (meshX_i_j * meshX_i_j / sigmax_squared +
                                     meshY_i_j * meshY_i_j / sigmay_squared);
            float pixVal = std::exp(pixVal2);
            float cosVal = pi_by_unfreq_by_2 * meshX_i_j;
            reffilter_i[j] = pixVal * std::cos(cosVal);
        }
    }

    cv::vector<cv::Mat> filters;

    for (int m = 0; m < 180 / angleInc; m++)
    {
        double angle = -(m * angleInc + 90);
        cv::Mat rot_mat =
            cv::getRotationMatrix2D(cv::Point((float)(refFilter.rows / 2.0F),
                                              (float)(refFilter.cols / 2.0F)),
                                    angle, 1.0);
        cv::Mat rotResult;
        cv::warpAffine(refFilter, rotResult, rot_mat, refFilter.size());
        filters.push_back(rotResult);
    }

    // Find indices of matrix points greater than maxsze from the image boundary
    int maxsze = szek;
    // Convert orientation matrix values from radians to an index value that
    // corresponds to round(degrees/angleInc)
    int maxorientindex = std::round(180 / angleInc);

    cv::Mat orientindex(rows, cols, CV_32FC1);

    int rows_maxsze = rows - maxsze;
    int cols_maxsze = cols - maxsze;

    for (int y = 0; y < rows; y++)
    {
        const auto *orientationImage_y = orientationImage.ptr<float>(y);
        auto *orientindex_y = orientindex.ptr<float>(y);
        for (int x = 0; x < cols; x++)
        {
            if (x > maxsze && x < cols_maxsze && y > maxsze && y < rows_maxsze)
            {
                validr.push_back(y);
                validc.push_back(x);
            }

            int orientpix = static_cast<int>(
                std::round(orientationImage_y[x] / M_PI * 180 / angleInc));

            if (orientpix < 0)
            {
                orientpix += maxorientindex;
            }
            if (orientpix >= maxorientindex)
            {
                orientpix -= maxorientindex;
            }

            orientindex_y[x] = orientpix;
        }
    }

    // Finally, do the filtering
    for (int k = 0; k < validr.size(); k++)
    {
        int r = validr[k];
        int c = validc[k];

        cv::Rect roi(c - szek - 1, r - szek - 1, meshX.cols, meshX.rows);
        cv::Mat subim(inputImage(roi));

        cv::Mat subFilter = filters.at(orientindex.at<float>(r, c));
        cv::Mat mulResult;
        cv::multiply(subim, subFilter, mulResult);

        if (cv::sum(mulResult)[0] > 0)
        {
            enhancedImage.at<float>(r, c) = 255;
        }
    }

    // Add a border.
    if (addBorder)
    {
        enhancedImage.rowRange(0, rows).colRange(0, szek + 1).setTo(255);
        enhancedImage.rowRange(0, szek + 1).colRange(0, cols).setTo(255);
        enhancedImage.rowRange(rows - szek, rows).colRange(0, cols).setTo(255);
        enhancedImage.rowRange(0, rows)
            .colRange(cols - 2 * (szek + 1) - 1, cols)
            .setTo(255);
    }

    return enhancedImage;
}

cv::Mat postProcessingFilter(const cv::Mat &inputImage)
{
    cv::Mat inputImageGrey;
    cv::Mat filter;

    if (inputImage.channels() != 1)
    {
        cvtColor(inputImage, inputImageGrey, COLOR_RGB2GRAY);
    }
    else
    {
        inputImageGrey = inputImage.clone();
    }

    // Blurring the image several times with a kernel 3x3
    // to have smooth surfaces
    for (int j = 0; j < blurringTimes; j++)
    {
        blur(inputImageGrey, inputImageGrey, cv::Size(3, 3));
    }

    // Canny detector to catch the edges
    Canny(inputImageGrey, filter, cannyLowThreshold,
          cannyLowThreshold * cannyRatio, kernelSize);

    // Use Canny's output as a mask
    cv::Mat processedImage(cv::Scalar::all(0));
    inputImageGrey.copyTo(processedImage, filter);

    cv::Mat element = cv::getStructuringElement(
        dilationType, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1),
        cv::Point(dilationSize, dilationSize));

    // Dilate the image to get the contour of the finger
    dilate(processedImage, processedImage, element);

    // Fill the image from the middle to the edge.
    floodFill(processedImage, cv::Point(filter.cols / 2, filter.rows / 2),
              cv::Scalar(255));

    return processedImage;
}

int minutiae_number = 0;

struct Minutia
{
    Point position;
    int type; // 1 for ridge ending, 2 for bifurcation
};

// Function to count the number of white neighbors around a pixel
int countNeighbors(const Mat &image, int x, int y)
{
    int count = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue; // Skip the center pixel
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && ny >= 0 && nx < image.cols && ny < image.rows && image.at<uchar>(ny, nx) == 255)
            {
                count++;
            }
        }
    }
    return count;
}

// Function to extract minutiae points using a 3x3 kernel
vector<Minutia> extractMinutiae(const Mat &skeleton)
{
    vector<Minutia> minutiae;

    for (int x = 0; x < skeleton.cols; x++)
    {
        for (int y = 0; y < skeleton.rows; y++)
        {
            if (skeleton.at<uchar>(y, x) == 255)
            { // If the pixel is part of the skeleton
                int neighbors = countNeighbors(skeleton, x, y);

                if (neighbors == 1)
                {
                    minutiae.push_back({Point(x, y), 1}); // Ridge ending
                }
                else if (neighbors == 4)
                {
                    minutiae.push_back({Point(x, y), 2}); // Bifurcation
                }
            }
        }
    }

    return minutiae;
}

// Function to calculate distance between two points
double distanceBetween(const Point &a, const Point &b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Function to filter out minutiae based on proximity and edge criteria
void filterMinutiae(vector<Minutia> &minutiae, double minDistance, int edgeThreshold, const Size &imageSize)
{
    // Remove minutiae too close to the edges
    minutiae.erase(remove_if(minutiae.begin(), minutiae.end(), [&](const Minutia &m)
                             { return m.position.x < edgeThreshold || m.position.y < edgeThreshold ||
                                      m.position.x > imageSize.width - edgeThreshold ||
                                      m.position.y > imageSize.height - edgeThreshold; }),
                   minutiae.end());

    // Remove minutiae too close to each other
    for (auto it1 = minutiae.begin(); it1 != minutiae.end(); ++it1)
    {
        for (auto it2 = minutiae.begin(); it2 != minutiae.end();)
        {
            if (it1 != it2 && distanceBetween(it1->position, it2->position) < minDistance)
            {
                it2 = minutiae.erase(it2);
            }
            else
            {
                ++it2;
            }
        }
    }
}

// Function to visualize minutiae
void visualizeMinutiae(const Mat &image, const vector<Minutia> &minutiae, Mat &output)
{
    cvtColor(image, output, COLOR_GRAY2BGR);
    for (const auto &m : minutiae)
    {
        Scalar color = (m.type == 1) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        circle(output, m.position, 2, color, 1);
    }
}

Rect findFingerprintArea(const Mat& image) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    gray.convertTo(gray, CV_8UC1);


    Mat binary;
    
    threshold(gray, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
    
    morphologyEx(binary,binary, MORPH_CLOSE,  cv::getStructuringElement(MORPH_CROSS, cv::Size(12, 12)));


    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    double maxArea = 0.0;
    int largestContourIndex = -1;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestContourIndex = i;
        }
    }

    if (largestContourIndex == -1) {
        return Rect();
    }

    Rect boundingRectangle = boundingRect(contours[largestContourIndex]);
    return boundingRectangle;
}

// Function to crop the image using the given rectangle
Mat cropImage(const Mat& image, const Rect& rect) {
    // Check if the rect is valid
    if (rect.width <= 0 || rect.height <= 0) {
        cout << "Invalid rectangle for cropping." << endl;
        return Mat();
    }

    // Crop the image
    Mat croppedImage = image(rect).clone();
    return croppedImage;
}


void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;    // north (pAbove)
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;    // south (pBelow)

    uchar *pDst;

    // initialize row pointers
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);

    for (y = 1; y < img.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);

        pDst = marker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for (x = 1; x < img.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) + 
                     (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) + 
                     (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                     (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }

    img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;         // convert to binary image

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    dst *= 255;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    // Input Image
    Mat image = imread(argv[1], IMREAD_COLOR);

   // imshow("Input Image", image);
    // Blurred Image
    Mat blurredImage;
    medianBlur(image, blurredImage, 3);

    // Check whether the input image is grayscale.
    // If not, convert it to grayscale.
    if (blurredImage.channels() != 1)
    {
        cvtColor(blurredImage, blurredImage, COLOR_RGB2GRAY);
    }

    //imshow("Blurred Image", blurredImage);

    imwrite("Blurred_image.jpg", blurredImage);

    // Perform normalization
    cv::Mat normalizedImage = normalize_image(blurredImage, 0, 1);

    //imshow("Normalized Image", normalizedImage);
    imwrite("Normalized_image.jpg", normalizedImage);
    // Calculate ridge orientation field
    Mat orientationImage = orient_ridge(normalizedImage);
    // imshow("Oriented Image", orientationImage);
    imwrite("orientation_image.jpg", orientationImage);

    cv::Mat freq = cv::Mat::ones(normalizedImage.rows, normalizedImage.cols,
                                 normalizedImage.type()) *
                   freqValue;

    cv::Mat enhancedImage =
        filter_ridge(normalizedImage, orientationImage, freq);

    // imshow("Enchaned Image", enhancedImage);
    imwrite("Enchaned_image.jpg", enhancedImage);

    Mat endResult(cv::Scalar::all(0));

    cv::Mat filter = postProcessingFilter(image);

    enhancedImage.copyTo(endResult, filter);
    //imshow("preproc Image", endResult);

    std::cout << "Type of the image  : " << enhancedImage.type()
              << std::endl;
    std::cout << "Type of the filter : " << filter.type()
              << std::endl;


   

       //Get Crop Coordinates
    Rect fingerprintArea = findFingerprintArea(endResult);
     imwrite("Enchaned_image_filtered.jpg", endResult);
    
    endResult.convertTo(endResult, CV_8UC1);


    // Otsu Binarization
    cv::Mat element_otsu = cv::getStructuringElement(MORPH_CROSS, cv::Size(3, 3));
    threshold(endResult, endResult, 0, 255, THRESH_OTSU);

    // Post Binarization thining
    
    element_otsu = cv::getStructuringElement(MORPH_CROSS, cv::Size(2, 2));
    cv::morphologyEx(endResult, endResult, MORPH_ERODE, element_otsu);
    imshow("otsu", endResult);

    imwrite("otsu_image.jpg", endResult);
    // skeletonization

    Mat skel;
    thinning(endResult,skel);
    imwrite("skeletonization.jpg", skel);
  

  
    imshow("skeletonization", skel);
    

    Mat croppedFingerprint = cropImage(skel, fingerprintArea);
    Mat croppedFirst = cropImage(image, fingerprintArea);

    // Minutiae Extraction
    vector<Minutia> minutiae = extractMinutiae(croppedFingerprint);
    // Visualize the result
    Mat output;
    visualizeMinutiae(croppedFingerprint, minutiae, output);
    imshow("unfiltered_Minutiae Points", output);
    imwrite("unfiltered_minutie.jpg", output);
    // Post-process to remove false minutiae
    filterMinutiae(minutiae, 10.0, 40, croppedFingerprint.size());

    // Visualize the result
    visualizeMinutiae(croppedFingerprint, minutiae, output);


    imshow("Minutiae Points", output);
    imwrite("minutie2.jpg", output);

    waitKey(0);
    return 0;
}
