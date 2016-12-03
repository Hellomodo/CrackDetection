#include <opencv2\opencv.hpp>
#include <list>

using namespace cv;
using namespace std;


class CrackDetection
{


public:
	CrackDetection(Mat& imgRaw,string &filename);

	~CrackDetection();

	Mat& GetImgCrackHighlight();

	float GetCracksScale();

private:
	Mat _imgRaw;

	Mat _imgCrackHighlight;

	string _filename;

	float _cracksaScale = 0;

	int FilterContours(Mat &imgSrc, Mat &imgDst, bool isSpecify, int numContours, double ratioThreshold );

	void GetHist(cv::Mat& srcImg, cv::Mat& dstImg);

	void CrackLocate(Mat &srcImg, Mat &binImg);

	float CrackAnalysis(Mat &srcImg, Mat &binImg);

}; 
