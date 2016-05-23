#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <string>

#define AnsQuantity 36           // Ans ��Ʈw   �V�m�˪O���
#define Quantity 50              // Train ��Ʈw ���Ѹ�� 
#define FQuantity 1080

const cv::Size2i normalize_size = cv::Size2i(128, 256);  //���W��


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
void sort(std::array<float, Quantity>& arr, int len)               // �w�j�Ƨ�
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

	// �����ܼ�
	int num = 0;
	int numTwo = 0;
	char Path[100];
	char PathTwo[100];

	// �ŧi�Ŷ�
	cv::Mat AnsDataMat(AnsQuantity, FQuantity, CV_32FC1);
	cv::Mat TraniningDataMat(Quantity, FQuantity, CV_32FC1);

	cv::Mat AOImg(AnsQuantity, FQuantity, CV_32FC1);
	cv::Mat TOImg(Quantity, FQuantity, CV_32FC1);
	//cv::Mat TImgtest(Quantity, FQuantity, CV_32FC1);
	//cv::Mat AImgtest(Quantity, FQuantity, CV_32FC1);

	//�Ĥ@���V�m�˪O��J
	for (int i = 0; i < AnsQuantity; i++)
	{
		//���

		sprintf_s(Path, sizeof(Path), "CharModelsAns/%03d.bmp", num);  //�зǼ˪O
	
		//��X����
		std::cout << "�V�m�˪O" << std::endl;
		printf("%s\n", Path);

		//Ū��
		cv::Mat AImg = cv::imread(Path, cv::IMREAD_GRAYSCALE);

		// ����䤣���
		if (AImg.empty())
		{
			std::cout << num << "Not Found!" << std::endl;
			exit(EXIT_FAILURE);
		}

		cv::resize(AImg, AImg, normalize_size);

		//��otsu
		//cv::threshold(AImg, AOImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//cv::imshow("AOImg",AOImg);

		//�N�ҧ��(HoughCircles)
		//cv::HoughCircles(AImg, AOImg, CV_HOUGH_GRADIENT, 1,5,100,10);

		//Binarization(AImg, AImgtest);  //�G�Ȥ�

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

		// �I�s
		printf("%d\n", descriptor.size());

		for (int j = 0; j < descriptor.size(); j++)
		{
			AnsDataMat.at<float>(i, j) = descriptor[j];
		}
		num++;
	}


	// �ĤG�����Ѽ˥���J
	for (int i = 0; i < Quantity; i++)
	{
		////�ܼ�
		//std::fstream Inside;
		//std::fstream Outside;
		////answer.txt
		////Ū txt��
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
		
		// ���
		sprintf_s(PathTwo, sizeof(PathTwo), "CharModels/1/1_Resized_%04d.bmp", i);  //���Ѽ˪O
        

		// ��X����
		std::cout << std::endl << "���Ѽ˥�";
		printf("%s\n", PathTwo);

		// Ū��
		cv::Mat TImg = cv::imread(PathTwo, cv::IMREAD_GRAYSCALE);

		// ����䤣���
		if (TImg.empty())
		{
			std::cout << "Not Found!" << std::endl;
			break;
		}

		cv::resize(TImg, TImg, normalize_size);
		// ��otsu
		//cv::threshold(TImg, TOImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// �N�ҧ��(HoughCircles)
		/*cv::HoughCircles(TImg, TOImg, CV_HOUGH_GRADIENT, 1,
		250, 200, 100);*/

		// HOG�Ѽ�
		CvSize win_size = TImg.size();
		CvSize block_size = cvSize(64, 128);
		CvSize stride_size = cvSize(32, 32);
		CvSize cell_size = cvSize(32, 32);

		// �ܼƫŧi
		int bins = 9;


		// HOG
		cv::HOGDescriptor hog = cv::HOGDescriptor(win_size, block_size, stride_size, cell_size, bins);

		std::vector<float> descriptors;
		CvSize winshiftsize = cvSize(1, 1);
		CvSize paddingsize = cvSize(0, 0);

		hog.compute(TImg, descriptors, winshiftsize, paddingsize);


		// ��X����
		// printf("%d\n", descriptors.size());


		//�ܼƩI�s
		std::vector<int> Ans;
		int Anss = 0;
		std::array<float, Quantity> distances = {};
		std::array<float, Quantity> SortDistances = {};
		float MinAns = 0;


		//Binarization(TImg, TImgtest);  //�G�Ȥ�

		//�ڦ��Z�����
		for (int j = 0; j < AnsQuantity; j++)
		{
			float distance = 0;

			for (int k = 0; k < descriptors.size(); k++)
			{
				distance += abs(AnsDataMat.at<float>(j, k) - descriptors[k]);
			}

			distances[j] = distance;                 // �N�ڦ��Z���۴�᪺��ƥ�J�}�C
		}
		for (int i = 0; i < sizeof(SortDistances) / sizeof(SortDistances[0]); i++)
		{
			SortDistances[i] = distances[i];
		}
		sort(SortDistances, AnsQuantity);           // �}�C�Ƨ�
		MinAns = SortDistances[0];    // ���o�}�C�̤p��   

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
		std::cout << "Anss(���̨ε��G) : " << Anss << std::endl;

		
		
	}
	system("pause");
	return 0;
}
