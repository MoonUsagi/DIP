#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>

#define AnsQuantity 36           // Ans 資料庫   訓練樣板資料
#define Quantity 50              // Train 資料庫 辨識資料 
#define FQuantity 1080

const cv::Size2i normalize_size = cv::Size2i(128, 256);  //正規化


void Binarization(cv::Mat binar, cv::Mat& binarout)
{
	for (int r = 1; r < binar.rows - 1; r++)
	{
		for (int c = 1; c < binar.cols - 1; c++)
		{
			if (binar.at<unsigned char>(r, c) >= 127)
			{
				binarout.at<unsigned char>(r, c) = 255;
			}
			else if (binar.at<unsigned char>(r, c)<127)
			{
				binarout.at<unsigned char>(r, c) = 0;
			}
		}
	}
}
void sort(std::array<float, Quantity>& arr, int len)               // 泡沫排序
{
	int i, j, temp;
	for (i = 0; i < len - 1; i++)
	{
		for (j = 0; j < len - 1 - i; j++)
		{
			if (arr[j] > arr[j + 1])
			{
				temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}

std::string s = "D:/MyProject-opencv/DIP/DIP/CharModels/6/";
int main(void)
{

	// 全域變數
	int num = 0;
	int numTwo = 0;
	char Path[100];
	char PathTwo[100];

	// 宣告空間
	cv::Mat AnsDataMat(AnsQuantity, FQuantity, CV_32FC1);
	cv::Mat TraniningDataMat(Quantity, FQuantity, CV_32FC1);

	cv::Mat AOImg(AnsQuantity, FQuantity, CV_32FC1);
	cv::Mat TOImg(Quantity, FQuantity, CV_32FC1);
	//cv::Mat TImgtest(Quantity, FQuantity, CV_32FC1);
	//cv::Mat AImgtest(Quantity, FQuantity, CV_32FC1);

	//第一次訓練樣板輸入
	for (int i = 0; i < AnsQuantity; i++)
	{
		//輸圖

		sprintf_s(Path, sizeof(Path), "CharModelsAns/%03d.bmp", num);  //標準樣板
	
		//輸出維度
		std::cout << "訓練樣板" << std::endl;
		printf("%s\n", Path);

		//讀圖
		cv::Mat AImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);

		// 防止找不到圖
		if (AImg.empty())
		{
			std::cout << num << "Not Found!" << std::endl;
			exit(EXIT_FAILURE);
		}

		cv::resize(AImg, AImg, normalize_size);

		//做otsu
		//cv::threshold(AImg, AOImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//cv::imshow("AOImg",AOImg);

		//霍夫找圓(HoughCircles)
		//cv::HoughCircles(AImg, AOImg, CV_HOUGH_GRADIENT, 1,5,100,10);

		//Binarization(AImg, AImgtest);  //二值化

		// HOG
		CvSize win_size = AImg.size();
		CvSize block_size = cvSize(64, 128);
		CvSize stride_size = cvSize(32, 32);
		CvSize cell_size = cvSize(32, 32);

		int bins = 9;

		cv::HOGDescriptor hog = cv::HOGDescriptor(win_size, block_size, stride_size, cell_size, bins);

		std::vector<float> descriptor;
		CvSize winshiftsize = cvSize(1, 1);
		CvSize paddingsize = cvSize(0, 0);

		hog.compute(AImg, descriptor, winshiftsize, paddingsize);

		// 呼叫
		printf("%d\n", descriptor.size());

		for (int j = 0; j < descriptor.size(); j++)
		{
			AnsDataMat.at<float>(i, j) = descriptor[j];
		}
		num++;
	}


	// 第二次辨識樣本輸入
	for (int i = 0; i < Quantity; i++)
	{
		////變數
		//std::fstream Inside;
		//std::fstream Outside;
		////answer.txt
		////讀 txt檔
		//Inside.open("Test.txt", std::ios::in);
		//while (Inside.getline(PathTwo, sizeof PathTwo));
		//{
		//	cv::Mat TImg = cv::imread(PathTwo, cv::IMREAD_GRAYSCALE);
		//	if (TImg.rows > 0, TImg.cols > 0)
		//	{

		//	}
		//	//Inside.read(PathTwo, 100);
		//	// if (PathTwo.rows > 0, PathTwo.cols < 0)
		//}
		
		// 輸圖
		sprintf_s(PathTwo, sizeof(PathTwo), "CharModels/1/1_Resized_%04d.bmp", i);  //辨識樣板
        

		// 輸出維度
		std::cout << std::endl << "辨識樣本";
		printf("%s\n", PathTwo);

		// 讀圖
		cv::Mat TImg = cv::imread(PathTwo, cv::IMREAD_GRAYSCALE);

		// 防止找不到圖
		if (TImg.empty())
		{
			std::cout << "Not Found!" << std::endl;
			break;
		}

		cv::resize(TImg, TImg, normalize_size);
		// 做otsu
		//cv::threshold(TImg, TOImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// 霍夫找圓(HoughCircles)
		/*cv::HoughCircles(TImg, TOImg, CV_HOUGH_GRADIENT, 1,
		250, 200, 100);*/

		// HOG參數
		CvSize win_size = TImg.size();
		CvSize block_size = cvSize(64, 128);
		CvSize stride_size = cvSize(32, 32);
		CvSize cell_size = cvSize(32, 32);

		// 變數宣告
		int bins = 9;


		// HOG
		cv::HOGDescriptor hog = cv::HOGDescriptor(win_size, block_size, stride_size, cell_size, bins);

		std::vector<float> descriptors;
		CvSize winshiftsize = cvSize(1, 1);
		CvSize paddingsize = cvSize(0, 0);

		hog.compute(TImg, descriptors, winshiftsize, paddingsize);


		// 輸出維度
		// printf("%d\n", descriptors.size());


		//變數呼叫
		std::vector<int> Ans;
		int Anss = 0;
		std::array<float, Quantity> distances = {};
		std::array<float, Quantity> SortDistances = {};
		float MinAns = 0;


		//Binarization(TImg, TImgtest);  //二值化

		//歐式距離比對
		for (int j = 0; j < AnsQuantity; j++)
		{
			float distance = 0;

			for (int k = 0; k < descriptors.size(); k++)
			{
				distance += abs(AnsDataMat.at<float>(j, k) - descriptors[k]);
			}

			distances[j] = distance;                 // 將歐式距離相減後的資料丟入陣列
		}
		for (int i = 0; i < sizeof(SortDistances) / sizeof(SortDistances[0]); i++)
		{
			SortDistances[i] = distances[i];
		}
		sort(SortDistances, AnsQuantity);           // 陣列排序
		MinAns = SortDistances[0];    // 取得陣列最小值   

		for (int index = 0; index < 36; index++)
		{
			if (distances[index] == MinAns)
			{
				Anss = index;
				Ans.push_back(index);
			}
		}


		for (size_t i = 0; i < Ans.size(); ++i) 
		{
			std::cout << Ans[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "Anss(比對最佳結果) : " << Anss << std::endl;

		
		
	}
	system("pause");
	return 0;
}
