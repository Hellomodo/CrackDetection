#include <opencv2\opencv.hpp>
#include <list>

using namespace cv;
using namespace std;


class CrackDetection
{


public:
	CrackDetection(Mat& imgRaw);

	~CrackDetection();

	Mat& GetImgCrackHighlight();

	float GetCracksScale();

private:
	Mat _imgRaw;

	Mat _imgCrackHighlight;

	float _cracksaScale = 0;

	int GetContourSpan(vector<Point> &contours);

	int FilterContours(Mat &imgSrc, Mat &imgDst, bool isSpecify, int numContours, double ratioThreshold );

	void GetHist(cv::Mat& srcImg, cv::Mat& dstImg);

	void CrackLocate(Mat &srcImg, Mat &binImg);

	float CrackAnalysis(Mat &srcImg, Mat &binImg);

}; 
