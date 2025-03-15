#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 定义棋盘格大小和每个方块的物理尺寸（单位为米）
const Size chessboardSize(6, 7); // 内部角点数
const float squareSize = 108.0f; 

void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

void StereoCalibrateAndRectify(const vector<string>& leftImages, const vector<string>& rightImages, Mat& cameraMatrixLeft, Mat& distCoeffsLeft,Mat& cameraMatrixRight,Mat& distCoeffsRight,Mat& R, Mat& T, Mat& Q) {
    vector<vector<Point2f>> imagePointsLeft, imagePointsRight;
    vector<vector<Point3f>> objectPoints;
    Size imageSize;

    // 准备棋盘格的3D点，假设Z坐标为0
    vector<Point3f> objectCorners;
    for (int i = 0; i < chessboardSize.height; ++i)
        for (int j = 0; j < chessboardSize.width; ++j)
            objectCorners.push_back(Point3f(j * squareSize, i * squareSize, 0));

    
   // cout <<  objectCorners << endl;

    for (int i = 0; i < leftImages.size(); ++i) {
        Mat imgLeft = imread(leftImages[i]);
        Mat imgRight = imread(rightImages[i]);

        if (imgLeft.empty() || imgRight.empty()) {
            cout << "无法读取图像: " << leftImages[i] << " 或 " << rightImages[i] << endl;
            continue;
        }

        // 进行灰度处理
        Mat grayLeft, grayRight;
        cvtColor(imgLeft, grayLeft, COLOR_BGR2GRAY);
        cvtColor(imgRight, grayRight, COLOR_BGR2GRAY);

        vector<Point2f> cornersLeft, cornersRight;
        bool foundLeft = findChessboardCorners(grayLeft, chessboardSize, cornersLeft);
        bool foundRight = findChessboardCorners(grayRight, chessboardSize, cornersRight);

        if (foundLeft && foundRight) {
            // 亚像素精度优化角点
            cornerSubPix(grayLeft, cornersLeft, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));
            cornerSubPix(grayRight, cornersRight, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));

            // 保存角点
            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);
            objectPoints.push_back(objectCorners);
        }

        // 获取图像大小
        if (imageSize == Size())
            imageSize = imgLeft.size();
    }

    // 标定单目相机
    Mat E, F;
    stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight,
                    cameraMatrixLeft, distCoeffsLeft,
                    cameraMatrixRight, distCoeffsRight,
                    imageSize, R, T, E, F,
                    0,
                    TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1e-5));

    
    cout << "左相机内参矩阵:\n" << cameraMatrixLeft << endl;
    cout << "右相机内参矩阵:\n" << cameraMatrixRight << endl;

    // 立体校正
    Mat R1, R2, P1, P2;
    stereoRectify(cameraMatrixLeft, distCoeffsLeft,
                  cameraMatrixRight, distCoeffsRight,
                  imageSize, R, T, R1, R2, P1, P2, Q);

    // 输出结果
    cout << "标定完成" << endl;

    // 生成去畸变和校正的映射表
    Mat map1Left, map2Left, map1Right, map2Right;
    initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, R1, P1, imageSize, CV_16SC2, map1Left, map2Left);
    initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, R2, P2, imageSize, CV_16SC2, map1Right, map2Right);


    // 对图像进行重映射去畸变并显示校正后的图像
    for (int i = 0; i < leftImages.size(); ++i) {
        Mat imgLeft = imread(leftImages[i]);
        Mat imgRight = imread(rightImages[i]);
        if (imgLeft.empty() || imgRight.empty()) {
            cout << "无法读取图像: " << leftImages[i] << " 或 " << rightImages[i] << endl;
            continue;
        }

        Mat undistortedLeft, undistortedRight;
        remap(imgLeft, undistortedLeft, map1Left, map2Left, INTER_LINEAR);
        remap(imgRight, undistortedRight, map1Right, map2Right, INTER_LINEAR);

    // 在校正后的图像上绘制水平线
    for (int y = 0; y < undistortedLeft.rows; y += 25) {
        // 在左图像上绘制水平线
        line(undistortedLeft, Point(0, y), Point(undistortedLeft.cols, y), Scalar(0, 255, 0), 1);
        // 在右图像上绘制水平线
        line(undistortedRight, Point(0, y), Point(undistortedRight.cols, y), Scalar(0, 255, 0), 1);
        
    }

        // 显示校正后的图像
        imshow("Left Rectified Image", undistortedLeft);
        imshow("Right Rectified Image", undistortedRight);
        waitKey(0);  // 按任意键继续
    }

    // 关闭所有窗口
    destroyAllWindows();

}


void StereoMatch(const string& leftimage, const string& rightimage, Mat& cameraMatrixLeft, Mat& distCoeffsLeft,Mat& cameraMatrixRight,Mat& distCoeffsRight,Mat& R, Mat& T, Mat& disp)
{
    // 读取图像
    Mat imgLeft = imread(leftimage);
    Mat imgRight = imread(rightimage);

    if (imgLeft.empty() || imgRight.empty()) {
        cout << "无法读取图像: " << leftimage << " 或 " << rightimage << endl;
        return;
    }
    
    // Mat re_R; 
    // invert(cameraMatrixRight,re_R);
    // Mat H =  cameraMatrixLeft * re_R;
    
    // Mat transformedRight;
    // warpPerspective(imgRight, transformedRight, H, imgRight.size());




    Size imageSize;
    imageSize = imgLeft.size();
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cameraMatrixLeft, distCoeffsLeft,
                  cameraMatrixRight, distCoeffsRight,
                  imageSize, R, T, R1, R2, P1, P2, Q);
    

    Mat map1Left, map2Left, map1Right, map2Right;
    initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, R1, P1, imageSize, CV_16SC2, map1Left, map2Left);
    initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, R2, P2, imageSize, CV_16SC2, map1Right, map2Right);
    
    Mat undistortedLeft, undistortedRight;
    remap(imgLeft, undistortedLeft, map1Left, map2Left, INTER_LINEAR);
    remap(imgRight, undistortedRight, map1Right, map2Right, INTER_LINEAR);

    // 在校正后的图像上绘制水平线
    for (int y = 0; y < undistortedLeft.rows; y += 25) {
        // 在左图像上绘制水平线
        line(undistortedLeft, Point(0, y), Point(undistortedLeft.cols, y), Scalar(0, 255, 0), 1);
        // 在右图像上绘制水平线
        line(undistortedRight, Point(0, y), Point(undistortedRight.cols, y), Scalar(0, 255, 0), 1);

        line(imgLeft, Point(0, y), Point(undistortedRight.cols, y), Scalar(0, 255, 0), 1);
        line(imgRight, Point(0, y), Point(undistortedRight.cols, y), Scalar(0, 255, 0), 1);
    }

    if (undistortedLeft.size() != undistortedRight.size()) {
        std::cout << "Error: Left and Right images have different sizes!" << std::endl;
    } else {
        std::cout << "Left and Right images have the same size." << std::endl;
    }

    // 显示校正后的图像
    imshow("Left Rectified Image", undistortedLeft);
    imshow("Right Rectified Image", undistortedRight);
    imshow("Left  Image", imgLeft);
    imshow("Right  Image", imgRight);

    waitKey(0);  // 按任意键继续

    
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,48,3);

    int pngChannels = imgLeft.channels();
    int winSize = 7;
    int numberOfDisparities = ((imageSize.width/8) + 15) & -16 ; 

    sgbm->setPreFilterCap(63);   //预处理滤波器截断值等
    sgbm->setBlockSize(winSize); //SAD窗口大小
    sgbm->setP1(8*pngChannels*winSize*winSize); //控制视差平滑度第一参数
    sgbm->setP2(32*pngChannels*winSize*winSize); //控制视差平滑度第二参数
    sgbm->setMinDisparity(0);  //最小视差
    sgbm->setNumDisparities(64); //视差搜索范围
    sgbm->setUniquenessRatio(5); //视差唯一性百分比
    sgbm->setSpeckleWindowSize(100); //检查视差连通区域变化度的窗口大小
    sgbm->setSpeckleRange(32); //视差变化阈值，当窗口内视差变化大于阈值时窗口内的视差清零
    sgbm->setDisp12MaxDiff(1);  //左右视差图最大容许差异，超过该阈值的视差值将被清零
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);  //采用全尺寸双通道动态编程算法

    sgbm->compute(undistortedLeft, undistortedRight, disp);

    int totalElements = disp.total();  // 获取视差图中元素的总数
    cout << "Total number of elements in disp: " << totalElements << endl;

    int zeroDisparities = countNonZero(disp == 0);
    cout << "Number of zero disparity pixels: " << zeroDisparities << endl;

    cout << "numberOfDisparities: " << numberOfDisparities << endl; 


    //cout << disp << endl;

}


void DepthGenerate(Mat& disp, Mat& T, Mat& cameraMatrixLeft, Mat& Q, const string& point_cloud_filename)
{

    int zeroDisparities = countNonZero(disp == 0);
    cout << "Number of zero disparity pixels: " << zeroDisparities << endl;

    //Mat depth; // 存储深度图
    Mat floatDisp;
    disp.convertTo(floatDisp, CV_32F, 1.0f / 16.0f);  // 将视差缩放回真实值

    Mat xyz;
    reprojectImageTo3D(floatDisp, xyz, Q, true); 

    saveXYZ(point_cloud_filename.c_str(), xyz);

    cv::Mat disp_normalized;
    cv::normalize(floatDisp , disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 显示视差图
    cv::imshow("Disparity Map", disp_normalized);
    cv::waitKey(0); // 等待按键

    cv::Mat mask = (floatDisp == -1); // 创建一个布尔掩码
    int countMinusOne = cv::countNonZero(mask); // 统计为 true 的元素个数
    std::cout << "Number of -1 elements: " << countMinusOne << std::endl;

    //cout << disp << endl;

    


    double f_x = cameraMatrixLeft.at<double>(0, 0);
    double baseline = norm(T); // T 是相机之间的平移向量

    //cv::Mat depth = cv::Mat::zeros(disp.size(), CV_64F); // CV_64F 表示浮点型深度


    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(disp, &minVal, &maxVal, &minLoc, &maxLoc);

    // cout << "Maxval: " << maxVal << endl;
    // cout << "minval: " << minVal << endl;




    // 创建一个 16 位无符号的整形深度图
    cv::Mat depth = cv::Mat::zeros(disp.size(), CV_16U);

    // 计算深度，并将其强制转换为 16 位无符号整型
    for (int y = 0; y < disp.rows; y++) {
        for (int x = 0; x < disp.cols; x++) {
            int disparity = disp.at<short>(y, x);
            if (disparity > 0) {
                // 计算深度（这里假设深度的单位为毫米）
                double depth_value = (f_x * baseline) / disparity;
                if (depth_value > 250)
                {
                    depth_value = 250.0;
                }
                //将深度转换为 16 位无符号整型，单位可以是毫米
                depth.at<uint16_t>(y, x) = static_cast<uint16_t>(depth_value);
            
            } else {
                depth.at<uint16_t>(y, x) = 58;  // 无效的视差点，设为 58
            }
        }
    }


    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(depth, &minVal, &maxVal, &minLoc, &maxLoc);

    cout << "Maxval: " << maxVal << endl;
    cout << "minval: " << minVal << endl;

    Mat depthDisplay;
    cv::normalize(depth, depthDisplay, 0, 65535, cv::NORM_MINMAX, CV_16U);

    // 保存深度图为 PNG 格式
    cv::imwrite("depth_image.png", depthDisplay);
    cv::imshow("Depth map", depthDisplay);

    cout << "baseline: " << baseline << endl;


    Mat depth_normalized;
    cv::normalize(depth , depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 应用颜色映射，将深度图转换为伪彩色图像
    cv::Mat depth_colormap;
    cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_JET);  // 使用 JET 颜色映射

    // 显示彩色深度图
    cv::imshow("Depth Colormap", depth_colormap);
    cv::waitKey(0);

    // 保存彩色深度图为 PNG 格式
    cv::imwrite("depth_colormap.png", depth_colormap);


    //cout << depth << endl;



}






int main() {
    // 自动从 images 文件夹中读取图像
    vector<string> leftImages, rightImages;
    glob("images_calib/left*.jpg", leftImages);  // 匹配 images 文件夹下所有 left 开头的图像
    glob("images_calib/right*.jpg", rightImages); // 匹配 images 文件夹下所有 right 开头的图像

    if (leftImages.size() != rightImages.size() || leftImages.empty()) {
        cerr << "左右图像数量不一致或文件夹为空" << endl;
        return -1;
    }

    string leftimage,rightimage;
    leftimage = "images_match/left0.jpg";
    rightimage = "images_match/right0.jpg";


    Mat cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, R, T, disp, Q;
    // 进行标定和立体校正
    StereoCalibrateAndRectify(leftImages, rightImages,cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight,R,T,Q);
    //cout << cameraMatrixLeft << endl;

    // 进行双目相机匹配与视差计算
    StereoMatch(leftimage,rightimage,cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight,R,T,disp);

    string point_cloud_filename = "output_point_cloud.xyz";
    DepthGenerate(disp, T, cameraMatrixLeft, Q, point_cloud_filename);

    return 0;
}
