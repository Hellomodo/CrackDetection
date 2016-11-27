// cracker_find.cpp : 定义控制台应用程序的入口点。
//

#include <opencv2\opencv.hpp>
#include <tchar.h>
#include <list>
#include <time.h> 
#include <io.h>
#include <math.h>

using namespace cv;
using namespace std;
RNG rng(123456);

void location(Mat &srcImg, Mat &binImg)
{
	vector< vector<Point> > contours;
	if (binImg.data)
	{
		findContours(binImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	}

	if (contours.size() > 0)
	{
		for (size_t i = 0; i < contours.size(); i++)
		{
			Rect maxRect;
			maxRect = boundingRect(contours[i]);
			rectangle(srcImg, maxRect, cv::Scalar(0, 0, 255));
		}
	}
}

double CrackAnalysis(Mat &srcImg, Mat &binImg)
{
	vector< vector<Point> > cracksList;
	int cracksProportion = 0, carcksAreaSum = 0;
	if (binImg.data)
	{
		findContours(binImg, cracksList, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	}

	if (cracksList.size() > 0)
	{
		drawContours(srcImg, cracksList, -1, Scalar(255,0,0), CV_FILLED, 8, vector<Vec4i>(), 2, Point());
	}

	vector< vector<Point> >::iterator it;
	for (it = cracksList.begin(); it != cracksList.end(); it++)
	{
		carcksAreaSum += contourArea(*it);
	}

	cracksProportion = carcksAreaSum / (srcImg.rows*srcImg.cols);
	return cracksProportion;
}

static inline bool ContoursSortByArea(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	return (contourArea(contour1) > contourArea(contour2));
}

int GetContourSpan( vector<Point> contours)
{
	int minX = 0x7FFFFFFF, minY = 0x7FFFFFFF;
	int maxX = 0, maxY = 0;
	for (vector<Point>::iterator it = contours.begin(); it != contours.end(); it++)
	{
		if (it->x > maxX)
			maxX = it->x;
		if (it->y > maxY)
			maxY = it->y;
		if (it->x < minX)
			minX = it->x;
		if (it->y < minY)
			minY = it->y;
	}
	return maxY - minY > maxX - minX ? maxY - minY : maxX - minX;
}

int FilterContours(Mat &imgSrc, Mat &imgDst, bool isSpecify = true, int numContours = 100, double ratioThreshold = 2)
{
	Mat imgTmp = Mat::zeros(imgSrc.size(), CV_8UC1);

	vector<vector<Point> > contoursToKeep;
	vector<vector<Point> > contoursAll;
	vector<Vec4i> hierarchy;
	
	/// 寻找轮廓
	findContours(imgSrc, contoursAll, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));

	if (contoursAll.size() <= 0)
	{
		return 0;
	}
	double thresholdArea = 0;
	double thresholdPARatio = ratioThreshold;

	sort(contoursAll.begin(), contoursAll.end(), ContoursSortByArea);

	double tmpSpan = GetContourSpan(contoursAll[0]);
	double tmpArea = contourArea(contoursAll[0]);
	//if the biggest shadow has the crack features
	if (tmpSpan > (imgSrc.cols+imgSrc.rows)*0.01
		&& (tmpSpan / 2)*(tmpSpan / 2)*3.14/ tmpArea > 2)
	{
		//isSpecify=false: so we just filter the img with no specifications
		if (false == isSpecify)
		{
			for (int i = 0; i < contoursAll.size() && i < numContours; i++)
			{
				tmpSpan = GetContourSpan(contoursAll[i]);
				if (tmpSpan >(imgSrc.rows + imgSrc.cols)*0.015)
				{
					contoursToKeep.push_back(contoursAll[i]);
				}
			}
		}
		else
		{
			for (int i = 0; i < contoursAll.size() && i < numContours; i++)
			{
				tmpArea = contourArea(contoursAll[i]);
				tmpSpan = GetContourSpan(contoursAll[i]);
				if ((tmpSpan / 2)*(tmpSpan / 2)*3.14 / tmpArea > ratioThreshold)
				{
					//删除面积小于设定值的轮廓  
					contoursToKeep.push_back(contoursAll[i]);
				}
			}
		}
	}

	
	drawContours(imgTmp, contoursToKeep, -1, Scalar(255), CV_FILLED, 8, vector<Vec4i>(), 2, Point());
	imgDst = imgTmp;
	return contoursToKeep.size();
}

void edgeEnhance(cv::Mat& srcImg, cv::Mat& dstImg)
{
	if (!dstImg.empty())
	{
		dstImg.release();
	}

	std::vector<cv::Mat> rgb;

	if (srcImg.channels() == 3)        // rgb image  
	{
		cv::split(srcImg, rgb);
	}
	else if (srcImg.channels() == 1)   // gray image  
	{
		rgb.push_back(srcImg);
	}

	// 分别对R、G、B三个通道进行边缘增强  
	for (size_t i = 0; i < rgb.size(); i++)
	{
		cv::Mat sharpMat8U;
		cv::Mat sharpMat;
		cv::Mat blurMat;

		// 高斯平滑  
		cv::GaussianBlur(rgb[i], blurMat, cv::Size(5, 5), 3, 3);

		// 计算拉普拉斯  
		cv::Laplacian(blurMat, sharpMat, CV_16S);

		// 转换类型  
		sharpMat.convertTo(sharpMat8U, CV_8U);
		cv::add(rgb[i], sharpMat8U, rgb[i]);
	}
	cv::merge(rgb, dstImg);
}

list<string> FindAllFile(const string &filespath)
{
	_finddata_t fileInfo;
	list<string> file_list;
	long findResult = _findfirst(filespath.c_str(), &fileInfo);
	if (-1L == findResult)
	{ 
		_findclose(findResult);
		return file_list;
	}
	
	do 
	{
		file_list.push_back(fileInfo.name);
	} while (0L == _findnext(findResult, &fileInfo));
	_findclose(findResult);
	return file_list;
}

void GetHist(cv::Mat& srcImg, cv::Mat& dstImg)
{
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	MatND hist;
	int channels[] = { 0 };

	calcHist(&srcImg, 1, channels, Mat(), // do not use mask  
		hist, 1, hist_size, ranges,
		true, // the histogram is uniform  
		false);

	double max_val;
	minMaxLoc(hist, 0, &max_val, 0, 0);
	int scale = 2;
	int hist_height = 256;
	Mat hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);
	for (int i = 0; i<bins; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val*hist_height / max_val);  //要绘制的高度  
		rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	dstImg = hist_img;
}

int _tmain(int argc, _TCHAR* argv[])
{

	clock_t start, finish;
	list<string> filelist;
	double totaltime;
	start = clock();
	filelist = FindAllFile("img_inputs\\*.jpg");
	filelist.splice(filelist.begin(),FindAllFile("img_inputs\\*.png"));

	for (list<string>::iterator it = filelist.begin(); it != filelist.end(); it++)
	{
		Mat imgRaw = imread(".\\img_samples\\" + *it);
		Mat imgEdgeEnhanceColor, imgEdgeEnhanceGray, imgEdgeBinary,edges,gray, biscuitMask;
		cvtColor(imgRaw, imgEdgeEnhanceColor, CV_RGB2GRAY);
		bilateralFilter(imgEdgeEnhanceColor, imgEdgeEnhanceGray, 3, 15,15);
		imwrite(".\\img_edgeEnhance\\" + *it, imgEdgeEnhanceGray);

		//Mat imgHist;
		//GetHist(imgEdgeEnhanceGray, imgHist);
		//imwrite(".\\img_hist\\" + *it, imgHist);

		int blockSize = 151;
		int constValue =15; //35
		int numCountours = 0;
		Mat elementDialate = getStructuringElement(MORPH_DILATE, Size(7, 7));
		Mat elementErode = getStructuringElement(MORPH_ERODE, Size(3, 3));
		adaptiveThreshold(imgEdgeEnhanceGray, imgEdgeBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
		imwrite(".\\img_adaptiveThreshold\\" + *it, imgEdgeBinary);
		
		numCountours = FilterContours(imgEdgeBinary, imgEdgeBinary, false);
		imwrite(".\\img_crackBinary\\" + *it, imgEdgeBinary);


		int iteration = 0;
		for(iteration = 0; numCountours >= 1; iteration++)
		{
			dilate(imgEdgeBinary, imgEdgeBinary, elementDialate);
			erode(imgEdgeBinary, imgEdgeBinary, elementErode);
			if (numCountours > 30)
			{
				numCountours = ceil(numCountours*0.6);
			}
			else if(numCountours > 15)
			{
				numCountours = ceil(numCountours*0.8);
			}
			else if(numCountours > 5)
			{
				numCountours --;
			}
			numCountours = FilterContours(imgEdgeBinary, imgEdgeBinary, true, numCountours, 0);

			if (numCountours < iteration)
			{
				break;
			}
		} 
		
		numCountours = FilterContours(imgEdgeBinary, imgEdgeBinary, true, numCountours, 2);
		CrackAnalysis(imgRaw, imgEdgeBinary);
		imwrite(".\\img_crackDetection\\" + *it, imgRaw);
		
	}

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\n此程序的运行时间为" << totaltime << "秒！" << endl;
	//system("pause");

	waitKey();

	return 0;
}

