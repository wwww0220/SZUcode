.
#include <opencv2\opencv.hpp>
#include <fstream>
#include <string>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#define MAXT 256	//rgb范围从0~255

using namespace cv;
using namespace std;


// 大津法
double dajinfa(Mat img, double t)
{
	//变量初始化
	int i = 0, j = 0;	//循环参数
	int grayarray[MAXT] = { 0 };	//灰度直方图,记录每个灰度值对应的像素个数
	int crow = img.rows;
	int ccol = img.cols;	//图像的行列数
	double sum1[MAXT] = { 0 };
	double Pofgray[MAXT] = { 0 };	//记录每一灰度值自身的概率
	double Pofsum[MAXT] = { 0 };	//每一项记录 0~该灰度值 的概率总和，用于得到 不同阈值 下 C0类 的 概率
	double Uofsum[MAXT] = { 0 };	//每一项记录 0~该灰度值 的均值总和，用于得到 不同阈值 下 C0类 的 均值
	double Uofsum1[MAXT] = { 0 };	//每一项记录 0~该灰度值 的均值总和，用于得到 不同阈值 下 C1类 的 均值
	double Oofsum[MAXT] = { 0 };	//每一项记录 C0 的方差，用于得到 不同阈值 下 C0类 的 方差
	double Oofsum1[MAXT] = { 0 };	//每一项记录 C0 的方差，用于得到 不同阈值 下 C1类 的 方差
	double sumofpixel = crow * ccol;	//像素总个数
	double flagsum = 0, flagofP = 0, flagofU = 0, flagofU1 = 0;	//用于概率和均值的累加操作
	double tflag = 0, qflag = 0;	//比较最优阈值的中间数
	double qw = 0, qb = 0; //记录类内和类间方差

	//统计直方图
	for (i = 0; i < crow; i++)
	{
		const uchar* rgbflag = img.ptr<uchar>(i);
		for (j = 0; j < ccol; j++)
		{
			grayarray[rgbflag[j]]++;	//通过rgbflag统计所有像素，进而统计图像直方图
		}
	}

	//概率和均值的计算
	for (i = 0; i < MAXT; i++)
	{
		Pofgray[i] = grayarray[i] / sumofpixel;	//某一灰度值出现概率 = 该灰度值像素个数 / 总灰度值
		
		sum1[i] = flagsum + grayarray[i];
		flagsum = sum1[i];
		
		Pofsum[i] = flagofP + Pofgray[i];	//flagofP初始为0，每次记录当前的概率总和，用于参与下次概率的加法
		flagofP = Pofsum[i];	

		for (j = 0; j <= i; j++)
		{
			Uofsum[i] += j * grayarray[j];
		}
		Uofsum[i] /= sum1[i];
	}
	
	for (i = 0; i < MAXT; i++)
	{
		for (j = i+1; j < MAXT; j++)
		{
			Uofsum1[i] += j * grayarray[j];	//与上面概率计算同理
		}
		Uofsum1[i] /= (sumofpixel - sum1[i]);
	}

	int sum = 0;
	for (i = 0; i < MAXT; i++)
	{
		//sum = 0;
		for (j = 0; j <= i; j++)
		{
			Oofsum[i] += grayarray[j] * (j - Uofsum[i]) * (j - Uofsum[i]);
			//sum += grayarray[j];
		}
		Oofsum[i] = Oofsum[i] / sum1[i];

		//sum = 0;
		for (j = i+1; j <= MAXT-1; j++)
		{
			Oofsum1[i] += grayarray[j] * (j - Uofsum1[i]) * (j - Uofsum1[i]);
			//sum += grayarray[j];
		}
		Oofsum1[i] = Oofsum1[i] / (sumofpixel - sum1[i]);
	}

	//最优阈值计算
	for (i = 0; i < MAXT; i++)
	{
		qw = Pofsum[i] * Oofsum[i] + (1 - Pofsum[i]) * Oofsum1[i]; //类内方差
		qb = Pofsum[i] * (Uofsum[i] - Uofsum[MAXT - 1]) * (Uofsum[i] - Uofsum[MAXT - 1]) + (1 - Pofsum[i]) * (Uofsum1[i] - Uofsum[MAXT - 1]) * (Uofsum1[i] - Uofsum[MAXT - 1]);
		qflag = qb / qw;
		if (qflag > tflag)
		{
			tflag = qflag;
			t = i;
		}
	}
	return t;
}

double diedaifa(Mat img)
{
	int grayarray[MAXT] = { 0 };	//灰度直方图,记录每个灰度值对应的像素个数
	double Tmax = 0, Tmin = 0, T0 = 0, Tk = 255 , t = 0;	//阈值
	double meanzf = 0, meanzb = 0;	//平均灰度
	int sumf = 0, sumb = 0;
	int i = 0, j = 0;	//循环
	int crow = img.rows;
	int ccol = img.cols;	//获得
	
	Point minLoc, maxLoc;

	minMaxLoc(img, &Tmin, &Tmax, &minLoc, &maxLoc);
	T0 = (Tmax + Tmin) / 2;	//初始阈值

	//统计直方图
	for (i = 0; i < crow; i++)
	{
		const uchar* rgbflag = img.ptr<uchar>(i);
		for (j = 0; j < ccol; j++)
		{
			grayarray[rgbflag[j]]++;	//通过rgbflag统计所有像素，进而统计图像直方图
		}
	}

	for (int k = 0; k < 256 || Tk - T0 > 1; k++)
	{
		if (k > 0)
			T0 = Tk;

		meanzf = 0;
		meanzb = 0;
		sumf = 0; 
		sumb = 0;

		for (i = 0; i < MAXT; i++)
		{
			if (i <= T0)
			{
				meanzf += i * grayarray[i];
				sumf += grayarray[i];
			}
			else
			{
				meanzb += i * grayarray[i];
				sumb += grayarray[i];
			}
		}
		meanzf /= sumf;
		meanzb /= sumb;
		Tk = (meanzf + meanzb) / 2;
	}
	t = Tk;
	return t;
}

int main()
{
	clock_t starttime, endtime;
	Mat a = imread("E:/BaiduNetdisk/大三/智能识别系统/实验1/车牌图像5张/车牌图像5张/xxx.bmp");
	if (a.empty())
	{
		cout << "图像文件路径错误" << endl;
		return 0;
	}
	Mat img = a;
	Mat imgflag;
	double t = 0, t1 = 0;	//阈值
	
	if(img.channels()>1)
		cvtColor(img, img, COLOR_BGR2GRAY);	//将图像转为灰度图

	starttime = clock();
	t=dajinfa(img, t);
	cout << "自己的大津法的阈值t=" << t << endl;
	endtime = clock();
	cout << "自己的大津法用时" << (double)(endtime - starttime) / CLOCKS_PER_SEC << "ms" << endl << endl;


	double Otsu = threshold(img, imgflag, 0, 255, THRESH_BINARY | THRESH_OTSU);
	cout << "官方的大津法的阈值t=" << Otsu << endl << endl;


	starttime = clock();
	t1 = diedaifa(img);
	cout << "自己的迭代法的阈值t=" << t1 << endl;
	endtime = clock();
	cout << "自己的迭代法用时" << (double)(endtime - starttime) / CLOCKS_PER_SEC << "ms" << endl << endl;

	namedWindow("img");
	imshow("img", img);
	waitKey(0);
	return(0);
}


/*视频播放
int main()
{
	//system("color F0");
	string flag;
	cout << "输入要播放的视频的地址：";
	cin >> flag;
	flag = flag.substr(1, flag.length() - 2);
	for (int i = 0; i < flag.length(); i++)
	{
		if (flag[i] == '\\')
		{
			flag[i] = '/';
		}
	}
	VideoCapture video(flag);
	if (video.isOpened())
	{
		cout << "视频中图像的宽度=" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
		cout << "视频中图像的高度=" << video.get(CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "视频帧率=" << video.get(CAP_PROP_FPS) << endl;
		cout << "视频的总帧数=" << video.get(CAP_PROP_FRAME_COUNT);
	}
	else
	{
		cout << "请确定视频文件地址是否正确" << endl;
		return -1;
	}
	while (true)
	{
		Mat frame;
		video >> frame;
		if (frame.empty())
			break;
		imshow("video", frame);
		waitKey(1000 / video.get(CAP_PROP_FPS));
	}
	waitKey();
	return 0;
}
*/

/*图片转单通道灰度图

void AlphaMat(Mat& mat)
{
	CV_Assert(mat.channels() == 4);
	for (int i = 0; i < mat.rows; ++i)
	{
		for (int j = 0; j < mat.cols; ++j)
		{

			Vec4b& bgra = mat.at<Vec4b>(i, j);
			bgra[0] = UCHAR_MAX;
			bgra[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
			bgra[2] = saturate_cast<uchar>((float(mat.cols - i)) / ((float)mat.rows) * UCHAR_MAX);
			bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2]));
		}
	}
}

int main()
{
	string flag;
	cout << "输入要转换图片的地址：";
	cin >> flag;
	flag = flag.substr(1, flag.length() - 2);
	for (int i = 0; i < flag.length(); i++)
	{
		if (flag[i]== '\\')
		{
			flag[i] = '/';
		}
	}

	Mat mat = imread(flag,0);

	imshow("tupian",mat);
	waitKey(0);
	//Mat mat(480, 640, CV_8UC4);
	//AlphaMat(mat);
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_JPEG_QUALITY);
	compression_params.push_back(1000);
	bool result = imwrite("D:/alpha.jpg", mat, compression_params);
	if (!result)
	{

		cout << "保存成jpg格式图像失败" << endl;
		return -1;
	}
	cout << "保存成功，保存位置为D:/alpha.jpg" << endl;
	return 0;
}*/

/*
读取、统计直方图、分离通道

int main()
{
	Mat img = imread("./1.jpg");


	int i = 0, j = 0;	//循环参数
	int grayarray[MAXT] = { 0 };	//灰度直方图,记录每个灰度值对应的像素个数
	int crow = img.rows;
	int ccol = img.cols;	//图像的行列数
	//统计直方图
	for (i = 0; i < crow; i++)
	{
		const uchar* rgbflag = img.ptr<uchar>(i);
		for (j = 0; j < ccol; j++)
		{
			grayarray[rgbflag[j]]++;	//通过rgbflag统计所有像素，进而统计图像直方图
		}
	}


	Mat img0, img1, img2;
	Mat imgs[3];
	//分离通道
	split(img, imgs);
	img0 = imgs[0];
	img1 = imgs[1];
	img2 = imgs[2];
	imshow("RGB_B", img0);
	imshow("RGB_G", img1);
	imshow("RGB_R", img2);
	imwrite("RGB_B.jpg", img0);
	imwrite("RGB_G.jpg", img1);
	imwrite("RGB_R.jpg", img2);



	// 应用Canny边缘检测
	Mat edges;
	double lowThreshold = 50;
	double highThreshold = 150;
	int kernelSize = 3;
	Canny(img, edges, lowThreshold, highThreshold, kernelSize);

	// 显示结果
	namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	imshow("Original Image", img);
	namedWindow("Edges", cv::WINDOW_AUTOSIZE);
	imshow("Edges", edges);
	imwrite("Edges.jpg", edges);



	namedWindow("测试");
	imshow("测试", img);
	waitKey(0);
	destroyAllWindows();
	return(0);

}

*/

/*
int main()
{
	//图像缩放，变马赛克
	Mat img = imread("./1.jpg", IMREAD_GRAYSCALE);
	Mat smallImg, bigImg0, bigImg1, bigImg2;
	resize(img, smallImg, Size(15, 15), 0, 0, INTER_AREA);
	resize(smallImg, bigImg0, Size(30, 30), 0, 0, INTER_NEAREST);	//最近邻插值
	resize(smallImg, bigImg1, Size(30, 30), 0, 0, INTER_LINEAR);	//双线性插值
	resize(smallImg, bigImg2, Size(30, 30), 0, 0, INTER_CUBIC);		//双三次插值
	namedWindow("smallImg", WINDOW_NORMAL);
	imshow("smallImg", smallImg);
	namedWindow("bigImg0", WINDOW_NORMAL);
	imshow("bigImg0", bigImg0);
	namedWindow("bigImg1", WINDOW_NORMAL);
	imshow("bigImg1", bigImg1);
	namedWindow("bigImg2", WINDOW_NORMAL);
	imshow("bigImg2", bigImg2);

	waitKey(0);
	return 0;
}
*/

/*

void savePixelsToFile(const Mat& image, const string& filename) {
	ofstream file(filename);
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			Vec3b pixel = image.at<Vec3b>(y, x);
			file << static_cast<int>(pixel[0]) << " "
				<< static_cast<int>(pixel[1]) << " "
				<< static_cast<int>(pixel[2]) << "\n";
		}
	}
	file.close();
}

int main() {
	// 读取图像
	Mat image = imread("./1.jpg");

	// 显示原图像
	imshow("Original Image", image);

	// 保存像素到 TXT 文件
	savePixelsToFile(image, "pixels.txt");

	// 定义感兴趣区域 (ROI)
	Rect roi(300, 450, 200, 200);
	Mat roiImage = image(roi);

	// 滤波操作
	Mat sobelX, sobelY, sobel, sharpened, gaussian;
	Sobel(image, sobelX, CV_64F, 1, 0);
	Sobel(image, sobelY, CV_64F, 0, 1);
	addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

	// 拉普拉斯锐化
	Mat laplacian;
	Laplacian(image, laplacian, CV_32F);
	//sharpened = image + laplacian;
	sharpened = laplacian;

	// 高斯滤波
	GaussianBlur(image, gaussian, Size(5, 5), 0);

	// 获取轮廓
	Mat gray, edges;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Canny(gray, edges, 100, 200);

	// 显示结果
	imshow("Sobel滤波", sobel);
	imshow("拉普拉斯锐化", sharpened);
	imshow("高斯滤波", gaussian);
	imshow("轮廓", edges);
	imshow("ROI", roiImage);

	waitKey(0);
	return 0;
}
*/