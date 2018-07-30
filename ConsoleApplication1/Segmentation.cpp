#include "Segmentation.h"
#include <stack>
using namespace cv;
using namespace std;

void segmentation(Mat &src, vector<vector<Point2f>> &groups) 
{
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			if (src.ptr<uchar>(row)[col] < 100) {
				Ptr<vector<Point2f>> group = new vector<Point2f>();
				checkPixel(src,*group,row, col);
				groups.push_back(*group);
			}
		}
	}

}

void checkPixel(Mat &src,vector<Point2f> &group,int row,int col){
	if (row < 0 || row >= src.rows || col < 0 || col >= src.cols) return;
	if (src.ptr<uchar>(row)[col] <150) {
		src.ptr<uchar>(row)[col] = 255;
		group.push_back(Point2f(col,row));
	}
	else {
		//src.ptr<uchar>(row)[col] = 99;
		return;
	}
	checkPixel(src, group, row - 1, col - 1);
	checkPixel(src, group, row - 1, col);
	checkPixel(src, group, row - 1, col + 1);
	checkPixel(src, group, row, col - 1);
	
	checkPixel(src, group, row, col + 1);
	checkPixel(src, group, row + 1, col - 1);
	checkPixel(src, group, row + 1, col);
	checkPixel(src, group, row + 1, col + 1);
}

Rect findRect(const vector<Point2f> &group) {
	auto i = group.begin();
	Point2f p = *i;
	int col_min = p.x, row_min = p.y, col_max = 0, row_max = 0;
	for (; i != group.end(); i++) {
		p = *i;
		if (col_min > p.x)	col_min = p.x;
		if (row_min > p.y)	row_min = p.y;
		if (col_max < p.x)	col_max = p.x;
		if (row_max < p.y)	row_max = p.y;
	}
	return Rect(Point2i(col_min, row_min), Point2i(col_max, row_max));
}

void findRects(const vector<vector<Point2f>> &groups, vector<cv::Rect> &rects) {
	for (auto i = groups.begin(); i != groups.end(); i++) {
		vector<Point2f> group = *i;
		rects.push_back(findRect(group));
	}
}
Mat resizeFraction(const vector<Point2f> &group, const int w,const int h) {

	static int index = 0;
	Rect rect = findRect(group);
	Mat num = Mat::zeros(Size(w,h), CV_8UC1);

	const double scale = std::min(w*0.9 / float(rect.width), h*0.9 / (float)rect.height);
	for (auto i = group.begin(); i != group.end(); i++) {
		Point2f p = *i;
		int x = (p.x - rect.x - rect.width / 2.0)*scale+w/2.0;
		int y = (p.y - rect.y - rect.height / 2.0)*scale + h / 2.0; 
		num.at<uchar>(y, x) = 255;
	}
	imwrite(format("E:\\img\\MNIST_data\\seg\\seg1_%d.png", index++), num);
	return num;
}