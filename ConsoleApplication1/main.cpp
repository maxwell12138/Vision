#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "Segmentation.h"
using namespace cv;
using namespace std;


int train_num = 20000;
int test_num = 1000;

int reverseDigit(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Mat readLabels(int opt) {
	int idx = 0;
	ifstream file;
	Mat img;
	int label_num;
	if (opt == 0)
	{
		cout << "\n Training...";
		file.open("dataset\\train-labels.idx1-ubyte");
		label_num = train_num;
	}
	else
	{
		cout << "\n Test...";
		file.open("dataset\\t10k-labels.idx1-ubyte");
		label_num = test_num;
	}
	// check file
	if (!file.is_open())
	{
		cout << "\n File Not Found!";
		return img;
	}
	/*
	byte 0 - 3 : Magic Number(Not to be used)
	byte 4 - 7 : Total number of labels in the dataset
	*/
	int magic_number = 0;
	int number_of_labels = 0;

	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseDigit(magic_number);
	file.read((char*)&number_of_labels, sizeof(number_of_labels));
	number_of_labels = reverseDigit(number_of_labels);

	cout << "\n No. of labels:" << number_of_labels << endl;
	Mat labels = Mat(label_num, 1, CV_32FC1);
	for (int i = 0; i<label_num; ++i)
	{
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		labels.at<float>(i, 0) = (float)temp;
	}
	return labels;
}

vector<Mat> readImages(int opt) {
	int idx = 0;
	ifstream file;
	vector<Mat> imgs;
	int img_num = 0;
	if (opt == 0)
	{
		cout << "\n Training...";
		file.open(".\\dataset\\train-images.idx3-ubyte", ios::binary);
		img_num = train_num;
	}
	else
	{
		cout << "\n Test...";
		file.open(".\\dataset\\t10k-images.idx3-ubyte", ios::binary);
		img_num = test_num;
	}

	// check file
	if (!file.is_open())
	{
		cout << "\n File Not Found!";
		return imgs;
	}
	/*
	byte 0 - 3 : Magic Number(Not to be used)
	byte 4 - 7 : Total number of images in the dataset
	byte 8 - 11 : rows of each image in the dataset
	byte 12 - 15 : cols of each image in the dataset
	*/

	int magic_number = 0;
	int number_of_images = 0;
	int height = 0;
	int width = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseDigit(magic_number);

	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseDigit(number_of_images);

	file.read((char*)&height, sizeof(height));
	height = reverseDigit(height);

	file.read((char*)&width, sizeof(width));
	width = reverseDigit(width);


	cout << "\n No. of images:" << number_of_images << endl;
	cout << " height of images:" << height << endl;
	cout << " width of images:" << width << endl;
	for (int i = 0; i < img_num; i++) {
		Mat train_image = Mat::zeros(height, width, CV_8UC1);
		for (int r = 0; r<height; ++r) {
			for (int c = 0; c<width; ++c){
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				train_image.at<uchar>(r, c) = (int)temp;
			}
		}
		imgs.push_back(train_image);
	}
	return imgs;
}

void getFeatures(vector<Mat> &train_images,Mat &output,int img_num) {
	//Mat features;
	HOGDescriptor hog(Size(28, 28), Size(20, 20), Size(4, 4), Size(10, 10), 8);
	for (int i = 0; i < img_num; i++) {
		vector<float> descriptor(100);
		Mat img = train_images[i];
		hog.compute(img, descriptor, Size(0, 0), Size(0, 0));
		if (i == 0) {
			//output = Mat::zeros(img_num, descriptor.size(), CV_32FC1);
		}
		for (int j = 0; j < descriptor.size(); j++) {
			output.at<float>(i, j) = descriptor[j];
		}
	}
}
void knnTrain() {
	vector<Mat> train_images = readImages(0);
	cout << " read all images" << endl;
	Mat train_labels = readLabels(0);
	cout << " read all labels" << endl;
	printf("\n read mnist train dataset successfully...\n");

	Mat trainLabels = train_labels(Range(0, train_num), Range::all());
	Mat trainFeatures;
	HOGDescriptor hog(Size(28, 28), Size(20, 20), Size(4, 4), Size(10, 10), 8);
	for (int i = 0; i < train_num; i++) {
		vector<float> descriptor(300);
		hog.compute(train_images[i], descriptor, Size(0, 0), Size(0, 0));
		if (i == 0) {
			trainFeatures = Mat::zeros(train_num, descriptor.size(), CV_32FC1);
		}
		for (int j = 0; j < descriptor.size(); j++) {
			trainFeatures.at<float>(i, j) = descriptor[j];
		}
	}
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->setDefaultK(10);
	knn->setIsClassifier(true);
	knn->train(tData);

	knn->save("knn.yml");
}

void testMnist() {
	vector<Mat> test_images = readImages(1);
	cout << " read all images" << endl;
	Mat test_labels = readLabels(1);
	cout << " read all labels" << endl;
	printf("\n read mnist test dataset successfully...\n");
	
	Ptr<ml::KNearest> knn = Algorithm::load<ml::KNearest>("knn.yml"); // KNN - 97%
	
	float correct = 0;
	HOGDescriptor hog(Size(28, 28), Size(20, 20), Size(4, 4), Size(10, 10), 8);
	
	Mat trainFeatures;
	for (int i = 0 ; i < test_num; i++) {
		int actual = test_labels.at<float>(i,0);
		//rect.y = i;
		vector<float> descriptor(288);
		Mat img = test_images[i];
		hog.compute(img, descriptor, Size(0, 0), Size(0, 0));
		if (i == 0) {
			trainFeatures = Mat::zeros(1, descriptor.size(), CV_32FC1);
		}
		for (int j = 0; j < descriptor.size(); j++) {
			trainFeatures.at<float>(0, j) = descriptor[j];
		}

		float predicted = knn->predict(trainFeatures);
		if (std::abs(predicted-actual)<0.05) {
			correct++;
		}
	}
	printf("\n recognize rate : %.2f \n", correct / test_num);
}
int predict(Mat img) {
	Ptr<ml::KNearest> knn = Algorithm::load<ml::KNearest>("knn.yml"); // KNN - 97%
	ifstream file;
	HOGDescriptor hog(Size(28, 28), Size(20, 20), Size(4, 4), Size(10, 10), 8);
	
	resize(img, img, Size(28, 28));
	vector<float> descriptor(288);
	hog.compute(img, descriptor, Size(0, 0), Size(0, 0));
	Mat trainFeatures = Mat::zeros(1, descriptor.size(), CV_32FC1);
	for (int j = 0; j < descriptor.size(); j++) {
		trainFeatures.at<float>(0, j) = descriptor[j];
	}
	int predicted = knn->predict(trainFeatures);
	return predicted;
}

void main(int argc,char** argv) {
	cout << " do you want to retrain the model(Y/N)?" << endl;
	char info;
	while (true) {
		cin >> info;
		if (info == 'Y' || info == 'y') {
			knnTrain();
			testMnist();
			break;
		}
		else if (info == 'N' || info == 'n') {
			break;
		}
		else {
			cout << "invalid input" << endl;
		}
	}

	string file;
	if (argc > 1) {
		file = argv[1];
	}
	else {
		file = "num.jpg";
	}

	cout << " Waiting..." << endl;
	Mat src = imread(file);
	if (src.empty()) {
		cout << " cannot find file. \n Press any key to exit..." << endl;
		char a;
		cin >> a;
		return;
	}
	cout << " Image loaded" << endl;

	double scales = std::min(src.rows / 300.0, src.cols / 600.0);
	resize(src, src, Size(src.cols/scales,src.rows/scales));

	Mat srcGray;
	cvtColor(src, srcGray, cv::COLOR_RGB2GRAY);
	Mat imgGray;
	threshold(srcGray, imgGray, 150, 255, CV_THRESH_BINARY);
	
	Ptr<vector<vector<Point2f>>> groups = new vector<vector<Point2f>>;
	segmentation(imgGray, *groups);
	cout << " Image devided" << endl;

	cout << " Draw bounding..." << endl;
	for each (vector<Point2f> var in *groups)
	{
		Rect rect = findRect(var);
		rectangle(src,rect,Scalar(0,255,0));

		Mat img = resizeFraction(var, 28, 28);
		int num = predict(img);
		std::string text = std::to_string(num);
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 1;
		int thickness = 2;
		int baseline;
		cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

		cv::Point origin;
		origin.x = rect.x + rect.width / 2 - text_size.width / 2;
		origin.y = rect.y + rect.height + text_size.height;
		cv::putText(src, text, origin, font_face, font_scale, cv::Scalar(0, 5, 0), thickness, 8, 0);
	}
	namedWindow("img", CV_WINDOW_AUTOSIZE);
	imshow("img", src);
	cout << " Image shown" << endl;
	imwrite("marked_img.jpg", src);
	cout << " Image saved" << endl;
	waitKey(0);
}