#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

string folder_path = "C:/Users/Miguel/Desktop/Cursos/Primavera_2018/Imagenes_avanzadas/Tarea_2/";

Mat harrisFilter(Mat input)
{
	Mat harris = Mat::zeros(input.rows, input.cols, CV_32FC1);
	// 0) Conversion a escala de grises
	Mat gray_input;
	cvtColor(input, gray_input, CV_BGR2GRAY);
	Mat gray_input_converted;
	gray_input.convertTo(gray_input_converted, CV_32FC1);
	// 1) Suavizado de imagen de entrada
	Mat smooth_input;
	GaussianBlur(gray_input_converted, smooth_input, Size(3, 3), 0, 0);
	// 2) Calculo de derivadas ix e iy
	Mat ix, iy;
	Sobel(smooth_input, ix, CV_32FC1, 1, 0);
	Sobel(smooth_input, iy, CV_32FC1, 0, 1);
	// 3) Calculo de momentos ixx, ixy, iyy
	//      ixx = ix*ix (elemento a elemento)
	//      ixy = ix*iy (elemento a elemento)
	//      iyy = iy*iy (elemento a elemento)
	Mat ixx, ixy, iyy;
	ixx = ix.mul(ix);
	ixy = ix.mul(iy);
	iyy = iy.mul(iy);
	// 4) Suavizado de momentos ixx, ixy, iyy
	Mat smooth_ixx, smooth_ixy, smooth_iyy;
	GaussianBlur(ixx, smooth_ixx, Size(3, 3), 0, 0);
	GaussianBlur(ixy, smooth_ixy, Size(3, 3), 0, 0);
	GaussianBlur(iyy, smooth_iyy, Size(3, 3), 0, 0);
	// 5) Calculo de harris como: det(m) - 0.04*Tr(m)^2, con:
	//      m = [ixx, ixy; ixy, iyy]
	//      det(m) = ixx*iyy - ixy*ixy;
	//      Tr(m) = ixx + iyy
	Mat det_m, Tr_m;
	det_m = smooth_ixx.mul(smooth_iyy) - smooth_ixy.mul(smooth_ixy);
	Tr_m = smooth_ixx + smooth_iyy;
	harris = det_m - 0.04 * (Tr_m.mul(Tr_m));
	// Normalizacion en rango 0-255
	Mat normalizedHarris;
	normalize(harris, normalizedHarris, 255, 0, NORM_MINMAX);
	Mat output;
	normalizedHarris.convertTo(output, CV_8UC1);
	return output;
}

vector<KeyPoint> getHarrisPoints(Mat harris, int val)
{
	vector<KeyPoint> points;
	double maxVal;
	Point maxLoc;
	int maskSize = 5;
	// Barrido de imagen mediante máscara
	for (int r = 0; r < harris.rows - maskSize + 1; r++)
	{
		for (int c = 0; c < harris.cols - maskSize + 1; c++)
		{
			Mat mask = harris(Rect(c, r, maskSize, maskSize));
			// Maximo local en máscara
			minMaxLoc(mask, NULL, &maxVal, NULL, &maxLoc);
			// Verificacion de unicidad y superación de umbral
			if (maxVal > val && maxLoc.x == maskSize / 2 && maxLoc.y == maskSize / 2)
				points.push_back(KeyPoint(c + maskSize / 2, r + maskSize / 2, maskSize));
		}
	}
	return points;
}

int main(void)
{
	Mat imleft = imread(folder_path + "left1.png");
	Mat imright = imread(folder_path + "right1.png");

	if(imleft.empty() || imright.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}

	// 1) Filtrado Harris
	Mat harrisleft, harrisright;	
	harrisleft = harrisFilter(imleft);
	harrisright = harrisFilter(imright);
	imwrite(folder_path + "harrisleft.jpg", harrisleft);
	imwrite(folder_path + "harrisright.jpg", harrisright);

	// 2) Puntos de Harris
	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> pointsleft = getHarrisPoints(harrisleft, 100);
	vector<KeyPoint> pointsright = getHarrisPoints(harrisright, 100);
	Mat impointsleft, impointsright;
	drawKeypoints(imleft, pointsleft, impointsleft);
	drawKeypoints(imright, pointsright, impointsright);
	imwrite(folder_path + "impointsleft.jpg", impointsleft);
	imwrite(folder_path + "impointsright.jpg", impointsright);

	// 3) Descriptores Locales
	Mat descrleft, descrright;
	orb->compute(imleft, pointsleft, descrleft);
	orb->compute(imright, pointsright, descrright);

	// 4) Matching de Descriptores
	Mat img_matches;
	vector<DMatch> matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(descrleft, descrright, matches);
	// 4.1) Limpieza de Matches
	sort(matches.begin(), matches.end());
	const float GOOD_MATCH_PERCENT = 0.15f;
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	drawMatches(imleft, pointsleft, imright, pointsright, matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite(folder_path + "img_matches.jpg", img_matches);

	// 5) Matriz de Transformacion de Imagen Derecha
	vector<Point2f> points1, points2;
	for (int i = 0; i < matches.size(); i++)
	{
		points1.push_back(Point2f(pointsleft[matches[i].queryIdx].pt));
		points2.push_back(Point2f(pointsright[matches[i].trainIdx].pt));
	}
	Mat H = findHomography(points2, points1, CV_RANSAC);

	// 6) Proyeccion de Imagen Derecha
	Mat imright_warp;
	warpPerspective(imright, imright_warp, H, Size(imleft.cols * 2, imleft.rows * 1.25));
	
	// 7) Reconstruccion de Imagen Panoramica
	Mat imfused(imright_warp.size(), CV_8UC3);
	Mat roi_left = imfused(Rect(0, 0, imleft.cols, imleft.rows));
	Mat roi_right = imfused(Rect(0, 0, imright_warp.cols, imright_warp.rows));
	imright_warp.copyTo(roi_right);
	imleft.copyTo(roi_left);
	imwrite(folder_path + "imFused.jpg", imfused);

	return 0; // Sale del programa normalmente
}
