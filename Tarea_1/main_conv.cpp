#define _DEBUG
#define _USE_MATH_DEFINES
// Instruciones:
// Dependiendo de la versión de opencv, deben usar los primeros dos includes (cv.h, highgui.h) o bien los dos includes siguientes (imgproc.hpp, highgui.hpp)

//#include <cv.h>
//#include <highgui.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

string folder_path = "C:/Users/Miguel/Desktop/Cursos/Primavera_2018/Imagenes_avanzadas/Tarea_1/src_alumnos/src_conv/";

void convolucion(Mat input, Mat mask, Mat output)
{
	float sum;
	for (int r=0; r<input.rows-mask.rows+1; r++)
	{
		for (int c=0; c<input.cols-mask.cols+1; c++)
		{
			sum = 0;
			for (int mask_r=0; mask_r<mask.rows; mask_r++)
			{
				for (int mask_c=0; mask_c<mask.cols; mask_c++)
				{
					sum += input.at<float>(r + mask_r, c + mask_c) * mask.at<float>(mask.rows - mask_r - 1, mask.cols - mask_c - 1);
				}
			}
			output.at<float>(r + mask.rows / 2, c + mask.cols / 2) = sum;
		}
	}
}

// No es necesario modificar esta funcion
Mat fft(Mat I)
{
	Mat padded;

	int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;

	merge(planes, 2, complexI);
	
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols/2;
	int cy = magI.rows/2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);
	
	Mat res;
	magI = magI*255;
	magI.convertTo(res, CV_8UC1);
	return res;
}

Mat GaussianBlur(int size, double sigma, bool laplacian)
{
	Mat kernel = Mat(size, size, CV_32FC1);
	double r, s = 2.0 * sigma * sigma;
	double sum = 0.0;

	// Kernel Generation
	for (int x = -size / 2; x <= size / 2; x++) {
		for (int y = -size / 2; y <= size / 2; y++) {
			r = sqrt(x * x + y * y);
			if (laplacian)
				kernel.at<float>(x + size / 2, y + size / 2) = - (1 - (r * r) / s) * (exp(-(r * r) / s)) / (M_PI * s * s / 2.0); // R*R
			else
				//kernel.at<float>(x + size / 2, y + size / 2) = (exp(-(r * r) / s)) / (M_PI * s);
				kernel.at<float>(x + size / 2, y + size / 2) = (exp(-(x * x + y * y) / 2.0 * sigma * sigma)) / (M_PI * 2.0 * sigma * sigma);
			sum += kernel.at<float>(x + size / 2, y + size / 2);
		}
	}

	// Kernel Normalization
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			kernel.at<float>(i, j) /= sum;

	return kernel;
}

Mat filtrar(Mat input, Mat mask, Mat output, string filter_name)
{
	cout << "Filter: " + filter_name << endl;
	// Conv 1D
	if ((mask.rows == 1) || (mask.cols == 1))
	{
		Mat output_aux = output.clone();
		convolucion(input, mask, output_aux);
		convolucion(output_aux, mask.t(), output);
	}
	// Conv 2D
	else
	{
		convolucion(input, mask, output);
	}

	Mat esp_out;
	esp_out = fft(output);
	imshow("spectrum_out", esp_out);
	imwrite(folder_path + "spectrum_out_" + filter_name + ".jpg", esp_out); // Grabar imagen

	output = abs(output);

	Mat last;

	output.convertTo(last, CV_8UC1);

	imshow("filtered", last);   // Mostrar imagen
	imwrite(folder_path + "filtered_" + filter_name + ".jpg", last); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV

	Mat concat;
	hconcat(last, esp_out, concat);
	return concat;
}

int main(void)
{
	Mat originalRGB = imread(folder_path + "figuras.png"); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}
	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);
	
	Mat input;
	original.convertTo(input, CV_32FC1);
	
	// Espectro Imagen Inicial
	Mat esp_in;
	esp_in = fft(input);
	imwrite(folder_path + "spectrum_in.jpg", esp_in); // Grabar imagen

	// Original concat
	Mat original_concat;
	hconcat(original, esp_in, original_concat);
	imwrite(folder_path + "original.jpg", original_concat); // Grabar imagen

	Mat output = input.clone();

	// 2.1. Filtro Pasa-bajos Recto
	Mat result_1;
	Mat mask_1 = Mat(3, 3, CV_32FC1);
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			mask_1.at<float>(i,j) = 1.0 / 9;
	result_1 = filtrar(input, mask_1, output, "LP_2D");

	// 2.2. Filtro Pasa-bajos Unidimensional
	Mat result_2;
	output = input.clone();
	Mat mask_2 = Mat(1, 3, CV_32FC1);
	for (int i = 0; i < 3; i++)
		mask_2.at<float>(i) = 1.0 / 3;
	result_2 = filtrar(input, mask_2, output, "LP_1D");

	// 2.3. Filtro Pasa-bajos Gaussiano
	Mat result_3;
	output = input.clone();
	Mat mask_3 = GaussianBlur(5, 1.0, false);
	result_3 = filtrar(input, mask_3, output, "LP_Gaussian_2D");

	// 2.4. Filtro Gaussiano Unidimensional
	Mat result_4;
	output = input.clone();
	Mat mask_4 = getGaussianKernel(5, 1.0, CV_32FC1).t();
	result_4 = filtrar(input, mask_4, output, "LP_Gaussian_1D");

	// Filtrados Pasa-bajos
	Mat aux_1; Mat aux_2; Mat aux_3;
	vconcat(result_1, result_2, aux_1);
	vconcat(result_3, result_4, aux_2);
	vconcat(aux_1, aux_2, aux_3);
	imwrite(folder_path + "LP_filters.jpg", aux_3); // Grabar imagen

	// 3.1. Filtro Pasa-altos Prewitt Vertical
	Mat result_5;
	output = input.clone();
	Mat mask_5 = Mat(3, 3, CV_32FC1);
	mask_5.at<float>(0, 0) = -1; mask_5.at<float>(0, 1) = 0; mask_5.at<float>(0, 2) = 1;
	mask_5.at<float>(1, 0) = -1; mask_5.at<float>(1, 1) = 0; mask_5.at<float>(1, 2) = 1;
	mask_5.at<float>(2, 0) = -1; mask_5.at<float>(2, 1) = 0; mask_5.at<float>(2, 2) = 1;
	mask_5 = mask_5 * 1.0 / 9;
	result_5 = filtrar(input, mask_5, output, "HP_Prewitt_vert");

	// 3.2. Filtro Pasa-altos Prewitt Horizontal
	Mat result_6;
	output = input.clone();
	Mat mask_6 = Mat(3, 3, CV_32FC1);
	mask_6.at<float>(0, 0) = -1; mask_6.at<float>(0, 1) = -1; mask_6.at<float>(0, 2) = -1;
	mask_6.at<float>(1, 0) = 0; mask_6.at<float>(1, 1) = 0; mask_6.at<float>(1, 2) = 0;
	mask_6.at<float>(2, 0) = 1; mask_6.at<float>(2, 1) = 1; mask_6.at<float>(2, 2) = 1;
	mask_6 = mask_6 * 1.0 / 9;
	result_6 = filtrar(input, mask_6, output, "HP_Prewitt_horz");

	// 3.3. Filtro Pasa-altos Laplaciano 3x3
	Mat result_7;
	output = input.clone();
	Mat mask_7 = Mat(3, 3, CV_32FC1);
	mask_7.at<float>(0, 0) = -1; mask_7.at<float>(0, 1) = -1; mask_7.at<float>(0, 2) = -1;
	mask_7.at<float>(1, 0) = -1; mask_7.at<float>(1, 1) = 8; mask_7.at<float>(1, 2) = -1;
	mask_7.at<float>(2, 0) = -1; mask_7.at<float>(2, 1) = -1; mask_7.at<float>(2, 2) = -1;
	mask_7 = mask_7 * 1.0 / 9;
	result_7 = filtrar(input, mask_7, output, "HP_Laplacian");

	// 3.4. Filtro Pasa-altos Gaussiano 5x5
	Mat result_8;
	output = input.clone();
	Mat mask_8 = GaussianBlur(5, 1.0, true);
	result_8 = filtrar(input, mask_8, output, "HP_Laplacian_Gaussian");

	// Filtrados Pasa-altos
	Mat aux_4; Mat aux_5; Mat aux_6;
	vconcat(result_5, result_6, aux_4);
	vconcat(result_7, result_8, aux_5);
	vconcat(aux_4, aux_5, aux_6);
	imwrite(folder_path + "HP_filters.jpg", aux_6); // Grabar imagen

	return 0; // Sale del programa normalmente
}
