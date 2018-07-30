#pragma once
#include <opencv2/opencv.hpp>
void segmentation(cv::Mat &src, std::vector<std::vector<cv::Point2f>> &groups);
void checkPixel(cv::Mat &src, std::vector<cv::Point2f> &group, const int row,const int col);
cv::Rect findRect(const std::vector<cv::Point2f> &group);
void findRects(const std::vector<std::vector<cv::Point2f>> &groups, std::vector<cv::Rect> &rects);
cv::Mat resizeFraction(const std::vector<cv::Point2f> &group, const int w, const int h);
