#include <io.h>
#include <opencv2\opencv.hpp>
#include <list>
#include "CrackDetection.h"

using namespace cv;
using namespace std;

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

int main(int argc, char* argv[])
{
	list<string> filelist;
	double totaltime;
	filelist = FindAllFile("img_inputs\\*.jpg");
	filelist.splice(filelist.begin(),FindAllFile("img_inputs\\*.png"));

	for (list<string>::iterator it = filelist.begin(); it != filelist.end(); it++)
	{
		Mat imgRaw = imread(".\\img_samples\\" + *it);
		CrackDetection crackInfo(imgRaw);
		imwrite(".\\img_crackDetection\\" + *it, crackInfo.GetImgCrackHighlight());
		cout << *it << "----->" << crackInfo.GetCracksScale() << endl;
	}

	return 0;
}

