#include"CrackDetection.h"

static inline int GetContourSpan(vector<Point> &contours)
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


static inline bool ContoursSortByArea(vector<cv::Point> &contour1, vector<cv::Point> &contour2)
{
	return (contourArea(contour1) > contourArea(contour2));
}

static inline bool ContoursSortBySpan(vector<cv::Point> &contour1, vector<cv::Point> &contour2)
{
	return (GetContourSpan(contour1) > GetContourSpan(contour2));
}

void CrackDetection::CrackLocate(Mat &srcImg, Mat &binImg)
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

float CrackDetection::CrackAnalysis(Mat &srcImg, Mat &binImg)
{
	vector< vector<Point> > cracksList;
	double cracksProportion = 0, carcksAreaSum = 0;
	if (binImg.data)
	{
		findContours(binImg, cracksList, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	}

	if (cracksList.size() > 0)
	{
		drawContours(srcImg, cracksList, -1, Scalar(255, 0, 0), CV_FILLED, 8, vector<Vec4i>(), 2, Point());
	}

	vector< vector<Point> >::iterator it;
	for (it = cracksList.begin(); it != cracksList.end(); it++)
	{
		carcksAreaSum += contourArea(*it);
	}

	cracksProportion = carcksAreaSum / (srcImg.rows*srcImg.cols);
	return cracksProportion;
}

void CrackDetection::GetHist(Mat& srcImg, Mat& dstImg)
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


int CrackDetection::FilterContours(Mat &imgSrc, Mat &imgDst, bool isSpecify = true, int numContours = 100, double ratioThreshold = 2)
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

	double thresholdPARatio = ratioThreshold;
	double thresholdArea = 0, thresholdSpan = 0;
	sort(contoursAll.begin(), contoursAll.end(), ContoursSortByArea);

	double tmpSpan = GetContourSpan(contoursAll[0]);
	double tmpArea = contourArea(contoursAll[0]);
	//if the biggest shadow has the crack features
	if (tmpSpan > (imgSrc.cols + imgSrc.rows)*0.01)
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

			thresholdArea += contourArea(contoursAll[0]);

			sort(contoursAll.begin(), contoursAll.end(), ContoursSortBySpan);

			thresholdSpan += GetContourSpan(contoursAll[0]);


			for (int i = 0; i < contoursAll.size() && i < numContours; i++)
			{
				tmpArea = contourArea(contoursAll[i]);
				tmpSpan = GetContourSpan(contoursAll[i]);
				if ((tmpSpan / 2)*(tmpSpan / 2)*3.14 / tmpArea > ratioThreshold
					&& tmpSpan / thresholdSpan > 0.1 && tmpArea /thresholdArea > 0.01)
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

CrackDetection::CrackDetection(Mat &imgRaw, string &filename)
{
	if (NULL == imgRaw.data)
		return;

	_imgRaw = imgRaw;

	Mat imgCracksGray;
	cvtColor(_imgRaw, imgCracksGray, CV_RGB2GRAY);
	Mat imgCracksFiltedGray;


	for (int i = 1; i < 9; i = i + 2)
	{
		bilateralFilter(imgCracksGray, imgCracksFiltedGray, i, i * 2, i / 2);
	}

	//bilateralFilter(imgCracksGray, imgCracksFiltedGray, 3, 15, 15);

	imwrite(".\\img_edgeEnhance\\" + filename, imgCracksFiltedGray);
	Mat imgHist;
	GetHist(imgCracksGray, imgHist);
	imwrite(".\\img_hist\\" + filename, imgHist);

	int blockSize = 151;
	int constValue = 15; 
	int numCountours = 0;
	Mat elementDialate = getStructuringElement(MORPH_DILATE, Size(7, 7));
	Mat elementErode = getStructuringElement(MORPH_ERODE, Size(3, 3));
	Mat imgCracksBinary;
	adaptiveThreshold(imgCracksFiltedGray, imgCracksBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
		
	imwrite(".\\img_adaptiveThreshold\\" + filename, imgCracksBinary);

	numCountours = FilterContours(imgCracksBinary, imgCracksBinary, false);
		
	imwrite(".\\img_crackFilter\\" + filename, imgCracksBinary);

	int iteration = 0;
	for (iteration = 0; numCountours >= 1; iteration++)
	{
		dilate(imgCracksBinary, imgCracksBinary, elementDialate);
		erode(imgCracksBinary, imgCracksBinary, elementErode);
		if (numCountours > 30)
		{
			numCountours = (int)ceil(numCountours*0.6);
		}
		else if (numCountours > 15)
		{
			numCountours = (int)ceil(numCountours*0.8);
		}
		else if (numCountours > 5)
		{
			numCountours--;
		}
		numCountours = FilterContours(imgCracksBinary, imgCracksBinary, true, numCountours, 0);

		if (numCountours < iteration || iteration > 4)
		{
			break;
		}
	}

	numCountours = FilterContours(imgCracksBinary, imgCracksBinary, true, numCountours, 2);

	imwrite(".\\img_crackBinary\\" + filename, imgCracksBinary);

	imgRaw.copyTo(_imgCrackHighlight);
	_cracksaScale = CrackAnalysis(_imgCrackHighlight, imgCracksBinary);

}

CrackDetection::~CrackDetection()
{
}

Mat& CrackDetection::GetImgCrackHighlight()
{
	return _imgCrackHighlight;
}

float CrackDetection::GetCracksScale()
{
	return _cracksaScale;
}