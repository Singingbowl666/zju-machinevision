#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 设置棋盘格尺寸
    int board_width = 7;
    int board_height = 6;
    float square_size = 108.0f;
    Size board_size(board_width, board_height);

    // 存储3D对象点和2D图像点
    vector<vector<Point3f>> objpoints;
    vector<vector<Point2f>> imgpoints;

    // 准备棋盘格的3D坐标
    vector<Point3f> objp;
    for (int i = 0; i < board_height; i++) {
        for (int j = 0; j < board_width; j++) {
            objp.push_back(Point3f(j*square_size, i*square_size, 0.0f));
        }
     }


    // 读取标定图像
    vector<String> images;
    glob("images/*.png", images);

    Mat img, gray;
    for (const auto& fname : images) {
        img = imread(fname);
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // 查找棋盘格角点
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, board_size, corners,CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            Size winSize(11, 11);
            find4QuadCornerSubpix(gray, corners, winSize);

            // 显示角点
            // drawChessboardCorners(img, board_size, Mat(corners), found);
            // imshow("Chessboard Corners", img);
            // waitKey(100);
            imgpoints.push_back(corners);
            objpoints.push_back(objp);
        }
    }
    // cout << objp << endl;
    cout << "Number of objpoints: " << objpoints.size() << endl;
    cout << "Number of imgpoints: " << imgpoints.size() << endl;

    cout << gray.size() << endl;
    // for (size_t i = 0; i < imgpoints.size(); ++i) {
    //     cout << "Image " << i + 1 << " corners:" << endl;
    //     for (size_t j = 0; j < imgpoints[i].size(); ++j) {
    //         cout << imgpoints[i][j] << endl;
    //     }
    //     cout << endl;
    // }

//     for (size_t i = 0; i < objpoints.size(); ++i) {
//     std::cout << "Object points set " << i+1 << ":\n";
//     for (size_t j = 0; j < objpoints[i].size(); ++j) {
//         std::cout << "(" << objpoints[i][j].x << ", " 
//                   << objpoints[i][j].y << ", " 
//                   << objpoints[i][j].z << ")\n";
//     }
//     std::cout << std::endl;
// }



    // 相机标定
    Mat cameraMatrix, distCoeffs, R, T;
    calibrateCamera(objpoints, imgpoints, gray.size(), cameraMatrix, distCoeffs, R, T);



    cout << cameraMatrix << endl;
    cout << distCoeffs << endl;


    // imageSize = gray.size()
    // 创建去畸变映射表

    Mat map1, map2;

    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, gray.size(), 1, gray.size(), 0),
        gray.size(), CV_16SC2, map1, map2);
    
    for (int i = 1; i <= 36; i++) {
        // 生成文件名 images/01.png 到 images/36.png
        stringstream ss;
        ss << "images/" << setfill('0') << setw(2) << i << ".png";
        string fileName = ss.str();
        // 读取图像
        Mat img = imread(fileName);
        if (img.empty()) {
            cout << "无法加载图像: " << fileName << endl;
            continue;
        }

        // 去畸变处理
        Mat img_undistorted;
        remap(img, img_undistorted, map1, map2, INTER_LINEAR);

        // 保存去畸变后的图像
        string outputFileName = "undistorted_" + fileName;
        imwrite(outputFileName, img_undistorted);
        cout << "已处理并保存: " << outputFileName << endl;

        // 显示原始图像和去畸变后的图像
        double scaleFactor = 0.5;
        Size newSize(img.cols * scaleFactor, img.rows * scaleFactor);


        Mat img_resized;
        resize(img, img_resized, newSize);
        Mat img_resized_undistorted;
        resize(img_undistorted, img_resized_undistorted, newSize);
        imshow("原始图像", img_resized);
        imshow("去畸变后的图像", img_resized_undistorted);
        cout << "按任意键显示下一个图像..." << endl;

        // 等待按键，按任意键继续
        waitKey(0); // 等待用户按键
    }

    // 关闭所有窗口
    destroyAllWindows();




    return 0;
}

