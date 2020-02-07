#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ml.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

string folder_path = "C:/Users/Miguel/Desktop/Cursos/Primavera_2018/Imagenes_avanzadas/Tarea_3/separated/";

// Uniform binary values lookup table
static int uniform[256] =
{
	1,2,3,4,5,0,6,7,8,0,0,0,9,0,10,11,12,0,0,0,0,0,0,0,13,0,0,0,14,0,
	15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,19,
	0,0,0,20,0,21,22,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,25,0,0,0,0,0,0,0,26,0,0,0,27,0,28,29,30,31,0,32,0,0,0,33,0,
	0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,36,37,38,0,39,0,0,0,40,0,0,0,0,0,0,0,41,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,42,43,44,0,45,0,0,0,46,0,0,0,0,0,0,0,47,48,49,0,50,
	0,0,0,51,52,53,0,54,55,56,57,58
};

Mat LBP_u2(Mat input)
{
	cvtColor(input, input, CV_BGR2GRAY);
	int windowSize = 3;
	Mat lbp_img = Mat(input.rows - windowSize + 1, input.cols - windowSize + 1, CV_8UC1);
	for (int r = 0; r < input.rows - windowSize + 1; r++)
	{
		for (int c = 0; c < input.cols - windowSize + 1; c++)
		{
			// Current image window
			Mat window = input(Rect(c, r, windowSize, windowSize));
			// Clockwise binary codification
			vector<int> binary_cod;
			uchar threshold = window.at<uchar>(1, 1);
			binary_cod.push_back(window.at<uchar>(0, 0) >= threshold);
			binary_cod.push_back(window.at<uchar>(0, 1) >= threshold);
			binary_cod.push_back(window.at<uchar>(0, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(1, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 1) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 0) >= threshold);
			binary_cod.push_back(window.at<uchar>(1, 0) >= threshold);
			// Uniform LBP value
			int lbp_value = 0;
			for (int i = 0; i < binary_cod.size(); i++)
				lbp_value += binary_cod.at(i) * pow(2, binary_cod.size() - 1 - i);
			lbp_img.at<uchar>(r, c) = uniform[lbp_value];
		}
	}
	return lbp_img;
}

Mat LBP_histogram(Mat input)
{
	Mat hist = Mat::zeros(1, 59, CV_8UC1);
	for (int r = 0; r < input.rows; r++)
	{
		for (int c = 0; c < input.cols; c++)
		{
			int ind = input.at<uchar>(r, c);
			hist.at<uchar>(0, ind) += 1;
		}
	}
	return hist;
}

Mat LBP_feature_vector(Mat input, int gridX, int gridY)
{
	Mat feature_vec, hist, window;
	int colsRange = input.cols / gridX;
	int rowsRange = input.rows / gridY;
	int rect_x, rect_y, rect_width, rect_height;
	for (int r = 0; r < gridY; r++)
	{
		for (int c = 0; c < gridX; c++)
		{
			rect_x = c * colsRange;
			rect_y = r * rowsRange;
			rect_width = (c + 1) * colsRange - 1;
			rect_height = (r + 1) * rowsRange -1;
			// Last windows
			if (r == gridY - 1)
				rect_height = input.rows - rect_y;
			if (c == gridX - 1)
				rect_width = input.cols - rect_x;
			// Current window
			window = input(Rect(rect_x, rect_y, rect_width, rect_height));
			// LBP histogram
			hist = LBP_histogram(window);
			// Histogram concatenation
			if (r == 0 && c == 0)
				feature_vec = hist;
			else
				hconcat(feature_vec, hist, feature_vec);
		}
	}
	return feature_vec;
}

int main(void)
{
	// Dataset Generation
	int n = 5; // Number of classes [2, 5].
	Mat img, lbp_img, feature_vec;
	vector<Mat> dataset;
	vector<String> classes = { "Asian", "Black", "Indian", "Others", "White" };
	for (int i = 0; i < n; i++)
	{
		Mat class_data;
		for (int j = 0; j < 200; j++)
		{
			img = imread(folder_path + classes[i] + "/" + to_string(j) + ".jpg");
			if (img.empty()) // Image not found
			{
				cout << "Image " + folder_path + classes[i] + "/" + to_string(j) + ".jpg" + "not found." << endl;
				return 1;
			}
			// LBP Image
			lbp_img = LBP_u2(img);
			// LBP Histogram Concatenated
			feature_vec = LBP_feature_vector(lbp_img, 2, 2);
			if (j == 0)
				class_data = feature_vec;
			else
				vconcat(class_data, feature_vec, class_data);
		}
		cout << "Dataset " + classes[i] + " (" << class_data.size << + ") generated!" << endl;
		dataset.push_back(class_data);
	}

	// Train-Test Dataset Separation
	Mat train, test, train_dataset, test_dataset, train_lab, test_lab, train_labels, test_labels;
	for (int i = 0; i < dataset.size(); i++)
	{
		train = dataset[i](Rect(0, 0, dataset[i].cols, dataset[i].rows * 0.7)).clone();
		test = dataset[i](Rect(0, dataset[i].rows * 0.7, dataset[i].cols, dataset[i].rows - dataset[i].rows * 0.7)).clone();
		train_lab = Mat(train.rows, 1, CV_8UC1);
		test_lab = Mat(test.rows, 1, CV_8UC1);
		train_lab.setTo(Scalar(i));
		test_lab.setTo(Scalar(i));
		if (i == 0)
		{
			train_dataset = train;
			test_dataset = test;
			train_labels = train_lab;
			test_labels = test_lab;
		}
		else
		{
			vconcat(train_dataset, train, train_dataset);
			vconcat(test_dataset, test, test_dataset);
			vconcat(train_labels, train_lab, train_labels);
			vconcat(test_labels, test_lab, test_labels);
		}
	}
	train_dataset.convertTo(train_dataset, CV_32F);
	test_dataset.convertTo(test_dataset, CV_32F);
	train_labels.convertTo(train_labels, CV_32SC1);
	test_labels.convertTo(test_labels, CV_32FC1);
	
	// Classifiers
	Mat prediction, diff, nonZeroCoordinates;
	float accuracy, misses;
	// SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100000, 0.01));
	svm->setC(0.0000001);
	svm->train(train_dataset, ROW_SAMPLE, train_labels);
	svm->predict(test_dataset, prediction);
	absdiff(prediction, test_labels, diff);
	diff.convertTo(diff, CV_8UC1);
	findNonZero(diff, nonZeroCoordinates);
	misses = nonZeroCoordinates.total();
	accuracy = 1 - misses / diff.size().height;
	cout << "SVM Accuracy: " + to_string(accuracy * 100) + "%" << endl;
	// Random Forest
	Ptr<ml::RTrees> randomForest = ml::RTrees::create();
	randomForest->train(train_dataset, ml::ROW_SAMPLE, train_labels);
	randomForest->predict(test_dataset, prediction);
	absdiff(prediction, test_labels, diff);
	diff.convertTo(diff, CV_8UC1);
	findNonZero(diff, nonZeroCoordinates);
	misses = nonZeroCoordinates.total();
	accuracy = 1 - misses / diff.size().height;
	cout << "Random Forest Accuracy: " + to_string(accuracy * 100) + "%" << endl;
	
	cvWaitKey(0);

	return 0;
}