#include <iostream>
#include "gcgraphMy.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc/imgproc_c.h>

#include <iostream>

using namespace std;
using namespace cv;

/*
This section shows how to use the library to compute
a minimum cut on the following graph :

*/
///

#include <stdio.h>

const int nDownSample = 1;
const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);
#define MASK_BG_COLOR   128
#define MASK_FG_COLOR   255
const Scalar FG_MASK_COLOR = Scalar(255, 255, 255);
const Scalar BG_MASK_COLOR = Scalar(128, 128, 128);

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void getBinMask(const Mat& comMask, Mat& binMask)
{
    if (comMask.empty() || comMask.type() != CV_8UC1)
        CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
        binMask.create(comMask.size(), CV_8UC1);
    binMask = comMask & 1;
}

static void showImageS2(const Mat& image, const string& winName)
{
    resizeWindow(winName.c_str(), image.cols / nDownSample, image.rows / nDownSample);
    imshow(winName, image);
}

class GCApplication
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName(const Mat& _image, const string& _winName);
    void showImage(int x, int y, int FgPoint);
    void mouseClick(int event, int x, int y, int flags, void* param);
    void graphConstruct(const Mat& img, GCGraphMy<double>& graph);
    void estimateSegmentation(GCGraphMy<double>& graph);
    int nextIter();
    int getIterCount() const { return iterCount; }
    void calSeedPHist(const Mat& img, const Mat& mask);
private:
    void setRectInMask();
    void fillSeedToMask(Mat& mask);
    void setLblsInMask(int x, int y, bool isFg);
    double calFgdPrioriCost(Vec3b &color);
    double calBgdPrioriCost(Vec3b &color);
    const string* winName;
    const Mat* image;
    Mat mask;
    Mat imgShowPg;
    Mat bgdModel, fgdModel;
    double FgPHist[3][256];
    double BgPHist[3][256];
    double gamma;
    double lambda;
    double beta;
    Mat leftW, upleftW, upW, uprightW;
    GCGraphMy<double> graphMy;
    uchar rectState, lblsState, prLblsState;
    bool isInitialized;
    Rect rect;
    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
};


void GCApplication::reset()
{
    if (!mask.empty())
    {
        mask.setTo(Scalar::all(GC_BGD));
        namedWindow("mask", 0);
    }
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();
    this->image->copyTo(imgShowPg);
    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
    if (_image.empty() || _winName.empty())
        return;
    image = &_image;
    winName = &_winName;
    mask.create(image->size(), CV_8UC1);
    reset();
}

void GCApplication::showImage(int x, int y, int FgPoint)
{
    static Point pre_pt(-1, -1);//初始坐标  
    if (image->empty() || winName->empty())
        return;
    pre_pt = Point(x, y);
    if (FgPoint == 1)
    {
        circle(imgShowPg, pre_pt, 3, BLUE, FILLED, CV_AA, 0);//划圆
        circle(mask, pre_pt, 3, FG_MASK_COLOR, FILLED, CV_AA, 0);//划圆
    }
    else if (FgPoint == 2)
    {
        circle(imgShowPg, pre_pt, 3, GREEN, FILLED, CV_AA, 0);//划圆
        circle(mask, pre_pt, 3, BG_MASK_COLOR, FILLED, CV_AA, 0);//划圆
    }

    showImageS2(imgShowPg, *(this->winName));
    showImageS2(mask, "mask");
}


void GCApplication::setRectInMask()
{
    assert(!mask.empty());
    mask.setTo(GC_BGD);
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols - rect.x);
    rect.height = min(rect.height, image->rows - rect.y);
    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void GCApplication::setLblsInMask(int x, int y, bool isFg)
{
    vector<Point> *bgpxls, *fgpxls;
    uchar bvalue, fvalue;
    bgpxls = &bgdPxls;
    fgpxls = &fgdPxls;
    Point p(x, y);
    //x,y就是原始图像中的，不需要上采样回去
    //p.x = p.x * nDownSample;//上采样回去
    //p.y = p.y * nDownSample;//上采样回去
    if (isFg)
    {
        fgpxls->push_back(p);
    }
    else
    {
        bgpxls->push_back(p);
    }

}

void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
    // TODO add bad args check
    switch (event)
    {

    case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
    {
        setLblsInMask(x, y, 1);
        showImage(x, y, 1);
        lblsState = SET;
    }
        break;
    case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
    {
        setLblsInMask(x, y, 0);
        showImage(x, y, 2);
        prLblsState = SET;
    }
        break;
    case EVENT_LBUTTONUP:
        lblsState = NOT_SET;
        break;
    case EVENT_RBUTTONUP:
        prLblsState = NOT_SET;
        break;
    case EVENT_MOUSEMOVE:
        if (lblsState != NOT_SET && flags & EVENT_FLAG_LBUTTON)
        {
            setLblsInMask(x, y, 1);
            showImage(x, y, 1);
        }
        else if (prLblsState != NOT_SET && flags & EVENT_FLAG_RBUTTON)
        {
            setLblsInMask(x, y, 0);
            showImage(x, y, 2);
        }
        break;
    default:
        lblsState = NOT_SET;
        prLblsState = NOT_SET;
        break;
    }
}
/*
Calculate beta - parameter of GrabCut algorithm.
beta = 1 / (2 * avg(sqr(|| color[i] - color[j] || )))
*/
static double calcBeta(const Mat& img)
{
    double beta = 0;
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x>0) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            if (y>0 && x>0) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            if (y>0) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            if (y>0 && x<img.cols - 1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2));

    return beta;
}

/*
Calculate weights of noterminal vertices of graph.
beta and gamma - parameters of GrabCut algorithm.
*/
static void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma)
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y, x) = 0;
            if (x + 1<img.cols && y - 1 >= 0) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y, x) = 0;
        }
    }
}

void GCApplication::calSeedPHist(const Mat& img, const Mat& mask)
{
    int nFgNum = 0;//
    int nBgNum = 0;//
    memset(&FgPHist[0][0], 0, 256 * 3 * sizeof(double));
    memset(&BgPHist[0][0], 0, 256 * 3 * sizeof(double));
  
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++)
    {
        for (p.x = 0; p.x < img.cols; p.x++)
        {
            uchar pMaskV = mask.at<uchar>(p);
            //背景像素值如直方图
            if (MASK_BG_COLOR == pMaskV)
            {
                Vec3b color = img.at<Vec3b>(p);
                nBgNum++;
                BgPHist[0][color[0]]++;
                BgPHist[1][color[1]]++;
                BgPHist[2][color[2]]++;
            }
            //前景像素值如直方图
            else if (MASK_FG_COLOR == pMaskV)
            {
                Vec3b color = img.at<Vec3b>(p);
                nFgNum++;
                FgPHist[0][color[0]]++;
                FgPHist[1][color[1]]++;
                FgPHist[2][color[2]]++;
            }
        }//
    }//

    nFgNum = nFgNum > 0 ? nFgNum : 1;//
    nBgNum = nBgNum > 0 ? nBgNum : 1;//

    //归一化并防止除0
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            FgPHist[i][j] = FgPHist[i][j] / nFgNum;
            FgPHist[i][j] = FgPHist[i][j] < 0.00001 ? 0.00001 : FgPHist[i][j];
            BgPHist[i][j] = BgPHist[i][j] / nBgNum;
            BgPHist[i][j] = BgPHist[i][j] < 0.00001 ? 0.00001 : BgPHist[i][j];
        }       
    }

}

double GCApplication::calFgdPrioriCost(Vec3b &color)
{
    double p = FgPHist[0][color[0]] * FgPHist[1][color[1]] * FgPHist[2][color[2]];
    return p;
}

double GCApplication::calBgdPrioriCost(Vec3b &color)
{
    double p = BgPHist[0][color[0]] * BgPHist[1][color[1]] * BgPHist[2][color[2]];
    return p;
}

void GCApplication::fillSeedToMask(Mat& mask)
{
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++)
    {
        for (p.x = 0; p.x < mask.cols; p.x++)
        {
            if (mask.at<uchar>(p) != MASK_BG_COLOR 
                && mask.at<uchar>(p) != MASK_FG_COLOR)
            {
                mask.at<uchar>(p) = 0;
            }
        }//
    }//
}


void GCApplication::graphConstruct(const Mat& img, GCGraphMy<double>& graph)
{
    gamma = 50;
    lambda = 1000;
    beta = calcBeta(*(this->image));

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

    int vtxCount = img.cols*img.rows,
        edgeCount = 2 * (4 * img.cols*img.rows - 3 * (img.cols + img.rows) + 2);

    fillSeedToMask(this->mask);
    calSeedPHist(img, this->mask);

    graph.create(vtxCount, edgeCount);
    Point p;
    double a = 1.5;
    for (p.y = 0; p.y < img.rows; p.y++)
    {
        for (p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(p) == 0)
            {
                fromSource = -a*log(calBgdPrioriCost(color));
                toSink = -a*log(calFgdPrioriCost(color));
            }
            else if (mask.at<uchar>(p) == MASK_BG_COLOR)
            {
                fromSource = 0;
                toSink = lambda;
            }
            else if (mask.at<uchar>(p) == MASK_FG_COLOR) // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // set n-weights,每个点只需要与左上4个点进行边连接即可,这样可以不重复的添加所有的N-8-edge
            if (p.x>0)
            {
                double w = leftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x>0 && p.y>0)
            {
                double w = upleftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y>0)
            {
                double w = upW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x<img.cols - 1 && p.y>0)
            {
                double w = uprightW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
        }
    }
}

/*
Estimate segmentation using MaxFlow algorithm
*/
void GCApplication::estimateSegmentation(GCGraphMy<double>& graph)
{
    graph.maxFlow();
    mask.setTo(GC_BGD);
    Point p;

    for (p.y = 0; p.y < mask.rows; p.y++)
    {
        for (p.x = 0; p.x < mask.cols; p.x++)
        {
            if (1 == graph.inSourceSegment(p.y*mask.cols + p.x /*vertex index*/))
            {
                mask.at<uchar>(p) = MASK_FG_COLOR;
            }
        }//
    }//
    showImageS2(mask, "mask");
    waitKey();
    destroyWindow("mask");
}


GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
    gcapp.mouseClick(event, x, y, flags, param);
}

int main()
{
    string filename = "/Users/pepperxzc/projects/graphcut/data/IMG-102223-0001.png";//分割图像路径
    Mat image = imread(filename, 1);
    if (image.empty())
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
    const string winName = "image";
	//缩放图像，避免图像太大，界面无法显示,且分辨率大的图，对本方法没有帮助
    resize(image, image, Size(image.cols / 3, image.rows / 3), 0, 0, INTER_LINEAR);
    namedWindow(winName, 0);
    resizeWindow(winName.c_str(), image.cols / nDownSample, image.rows / nDownSample);
    gcapp.setImageAndWinName(image, winName);

    setMouseCallback(winName, on_mouse, 0);

    imshow(winName, image);
    waitKey();

    GCGraphMy <double>stGraphMy;

    gcapp.graphConstruct(image, stGraphMy);

    gcapp.estimateSegmentation(stGraphMy);
    destroyWindow(winName);

    //gcapp.setImageAndWinName(image, winName);
    //gcapp.showImage();

    system("pause");

    return 0;
}


///

