#define _DEBUG
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

string folder_path = "C:/Users/Miguel/Desktop/Cursos/Primavera_2018/Imagenes_avanzadas/Tarea_1/src_alumnos/src_histeq/";

void ecualizar(Mat input, Mat output)
{
	// Histogram
	float hist[256];
	for (int i=0; i<256; i++)
		hist[i] = 0;
	for (int r=0; r<input.rows; r++)
	{
		for (int c=0; c<input.cols; c++)
		{
			int ind = input.at<unsigned char>(r,c);
			hist[ind] = hist[ind] + 1.0/input.rows/input.cols;
		}
	}
	// Lookup Table
	int lookup[256];
	float sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += hist[i];
		lookup[i] = sum * 255 + 0.5;
	}
	// Image Mapping
	for (int r = 0; r < input.rows; r++)
	{
		for (int c = 0; c < input.cols; c++)
		{
			output.at<unsigned char>(r, c) = lookup[input.at<unsigned char>(r, c)];
		}
	}
	return;
}

int ecualizar_img(string filename)
{
	cout << "Image: " + filename << endl;
	Mat originalRGB = imread(folder_path + filename); //Leer imagen

	if (originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}

	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);

	Mat output = Mat::zeros(original.rows, original.cols, CV_8UC1);
	ecualizar(original, output);

	imshow("original", original);   // Mostrar imagen
	imshow("ecualizado", output);   // Mostrar imagen
	imwrite(folder_path + filename + "_ecualizado.jpg", output); // Grabar imagen

	Mat concat;
	hconcat(original, output, concat);
	imwrite(folder_path + filename + "_resultado.jpg", concat); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV
	return 0;
}

int main(void)
{
	ecualizar_img("mala_ilum.jpg");
	ecualizar_img("agua.png");
	ecualizar_img("casa.png");
	ecualizar_img("constr.png");
	ecualizar_img("corteza.png");
	ecualizar_img("termal.png");
	return 0; // Sale del programa normalmente
}
