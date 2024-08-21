
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

Mat loadAndPreprocessImage(const string& imagePath) {
    Mat img = imread(imagePath);
    Mat croppedImg = cropImage(img, 0.10, 0.10, 0.05, 0.05);
    Mat resizedImg;
    resize(croppedImg, resizedImg, Size(256, 500));
    Mat grayImage;
    cvtColor(resizedImg, grayImage, COLOR_BGR2GRAY);
    Mat blurredImage;
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);
    return blurredImage;
}

Mat cropImage(const Mat& image, double leftPct, double rightPct, double topPct, double bottomPct) {
    int height = image.rows;
    int width = image.cols;
    int left = static_cast<int>(width * leftPct);
    int right = static_cast<int>(width * rightPct);
    int top = static_cast<int>(height * topPct);
    int bottom = static_cast<int>(height * bottomPct);
    return image(Range(top, height - bottom), Range(left, width - right));
}

Mat computeGradients(const Mat& blurredImage) {
    Mat gradientX, gradientY;
    Scharr(blurredImage, gradientX, CV_64F, 1, 0);
    Scharr(blurredImage, gradientY, CV_64F, 0, 1);
    Mat gradientMagnitude;
    magnitude(gradientX, gradientY, gradientMagnitude);
    return gradientMagnitude;
}

vector<Rect> findContours(const Mat& gradientMagnitude, double threshold, double minAreaThreshold) {
    Mat highContrastAreas;
    threshold(gradientMagnitude, highContrastAreas, threshold, 255, THRESH_BINARY);
    highContrastAreas.convertTo(highContrastAreas, CV_8U);
    vector<vector<Point>> contours;
    findContours(highContrastAreas, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Rect> rectangles;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > minAreaThreshold) {
            Rect boundingBox = boundingRect(contour);
            rectangles.push_back(boundingBox);
        }
    }
    return rectangles;
}

vector<Rect> nonMaxSuppression(const vector<Rect>& rectangles, double overlapThresh) {
    if (rectangles.empty()) {
        return {};
    }

    vector<Rect> pick;
    vector<int> idxs(rectangles.size());
    iota(idxs.begin(), idxs.end(), 0);
    sort(idxs.begin(), idxs.end(), [&](int i1, int i2) {
        return rectangles[i1].br().y < rectangles[i2].br().y;
    });

    while (!idxs.empty()) {
        int last = idxs.size() - 1;
        int i = idxs[last];
        pick.push_back(rectangles[i]);
        vector<int> suppress;
        suppress.push_back(last);

        for (int pos = 0; pos < last; ++pos) {
            int j = idxs[pos];
            int xx1 = max(rectangles[i].x, rectangles[j].x);
            int yy1 = max(rectangles[i].y, rectangles[j].y);
            int xx2 = min(rectangles[i].br().x, rectangles[j].br().x);
            int yy2 = min(rectangles[i].br().y, rectangles[j].br().y);
            int w = max(0, xx2 - xx1 + 1);
            int h = max(0, yy2 - yy1 + 1);
            double overlap = (w * h) / static_cast<double>(rectangles[j].area());

            if (overlap > overlapThresh) {
                suppress.push_back(pos);
            }
        }

        for (int pos = suppress.size() - 1; pos >= 0; --pos) {
            idxs.erase(idxs.begin() + suppress[pos]);
        }
    }
    return pick;
}

void drawRectangles(Mat& image, const vector<Rect>& rectangles, double minRequiredArea, double maxAspectRatio) {
    for (const auto& rect : rectangles) {
        double aspectRatio = static_cast<double>(rect.width) / rect.height;
        double area = rect.width * rect.height;
        if (aspectRatio <= maxAspectRatio && area > minRequiredArea) {
            rectangle(image, rect, Scalar(0, 255, 0), 2);
            int centerX = rect.x + rect.width / 2;
            int centerY = rect.y + rect.height / 2;
            circle(image, Point(centerX, centerY), 1, Scalar(0, 0, 255), -1);
            double yNormalized = 1.0 - static_cast<double>(centerY) / 500.0;
            drawText(image, rect, yNormalized);
        }
    }
}

void drawText(Mat& image, const Rect& rect, double yNormalized) {
    string text = to_string(yNormalized).substr(0, 4);
    int baseline = 0;
    int font = FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    Size textSize = getTextSize(text, font, scale, thickness, &baseline);
    int textX = rect.x + (rect.width - textSize.width) / 2;
    int textY = rect.y - textSize.height - 2;
    putText(image, text, Point(textX, textY), font, scale, Scalar(255, 0, 0), thickness);
}

void mainProcess(const string& imagePath) {
    Mat resizedImg, blurredImage;
    blurredImage = loadAndPreprocessImage(imagePath);
    Mat gradientMagnitude = computeGradients(blurredImage);

    double initialThreshold = 50;
    double initialMinAreaThreshold = 200;
    int minRequiredRectangles = 7;
    double minRequiredArea = 250;
    double maxAspectRatio = 3;

    vector<Rect> rectangles;
    int numRectangles = 0;
    while (numRectangles < minRequiredRectangles) {
        rectangles = findContours(gradientMagnitude, initialThreshold, initialMinAreaThreshold);
        rectangles = nonMaxSuppression(rectangles, 0.2);
        drawRectangles(resizedImg, rectangles, minRequiredArea, maxAspectRatio);
        numRectangles = rectangles.size();
        initialMinAreaThreshold -= 100;
    }

    imwrite("Output.jpg", resizedImg);
}

int main() {
    mainProcess("121.jpeg");
    return 0;
}
