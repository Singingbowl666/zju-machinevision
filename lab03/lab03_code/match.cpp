#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void featureMatchingAndPoseEstimation(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& cameraMatrix) {
    // 1. 特征点检测
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    cout << keypoints1.size() << endl;
    cout << keypoints2.size() << endl;

    // 绘制关键点
    Mat img_keypoints1, img_keypoints2;
    drawKeypoints(img1, keypoints1, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img2, keypoints2, img_keypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // 显示结果图像
    imshow("Keypoints Image 1", img_keypoints1);
    imshow("Keypoints Image 2", img_keypoints2);

    waitKey(0);

    // 2. 特征点匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
   
    Mat result1;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, result1);
    // 显示结果
    imshow("Matches", result1);
    waitKey(0);

    // 3. 筛选匹配点
    double maxDist = 0, minDist = 100;
    for (const auto& match : matches) {
        double dist = match.distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : matches) {
        if (match.distance <= std::max(2 * minDist, 30.0)) {
            goodMatches.push_back(match);
        }
    }

    // 4. 计算位姿
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    
    // 假设 mask 是存储每个匹配点是否为内点的布尔向量
    std::vector<uchar> mask; 
    // 存储内点的匹配点
    std::vector<cv::DMatch> Inliers;
    Mat distCoeffs;
    // 计算基础矩阵或本质矩阵
    cv::Mat E = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC);

/////////////////////////////////////////////////////
    // 最小匹配点数
    const int minMatchCount = 5;
    if (points1.size() < minMatchCount) {
        std::cout << "Not enough matches to compute essential matrix!" << std::endl;
    }
    // 计算 RANSAC 迭代次数
    double confidence = 0.99; // 期望置信度
    double p = 0.8; // 估计内点比例
    int n = 5; // 本质矩阵所需的最小点对数

    // 计算 RANSAC 迭代次数
    int N = std::log(1 - confidence) / std::log(1 - std::pow(p, n));
    std::cout << "Required RANSAC iterations: " << N << std::endl;
/////////////////////////////////////////////////////

    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, cameraMatrix, R, t,mask);

    // 计算内点比例
    int inlierCount = 0;
    int totalMatches = mask.size(); // 总匹配点数

    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            inlierCount++; // 统计内点
        }
    }

    // 计算内点比例
    double inlierRatio = static_cast<double>(inlierCount) / totalMatches;

    std::cout << "Number of inliers: " << inlierCount << std::endl;
    std::cout << "Total matches: " << totalMatches << std::endl;
    std::cout << "Inlier ratio: " << inlierRatio << std::endl;

    for (size_t i = 0; i < matches.size(); i++) {
    // 如果当前匹配点是内点
    if (mask[i]) {
        Inliers.push_back(matches[i]); // 添加到内点列表中
    }
    }

    cout << "numbers of matches points " << matches.size() << endl;
    cout << "numbers of Inlier points " << Inliers.size() << endl;
   
    Mat result2;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, Inliers, result2);
    // 显示结果
    imshow("Matches", result2);
    waitKey(0);


    // 输出结果
    std::cout << "旋转矩阵 R:\n" << R << std::endl;
    std::cout << "平移向量 t:\n" << t << std::endl;

    Mat result;
    vconcat(result1, result2, result);
    imshow("Matches", result);
    waitKey(0);


}

int main(int argc, char** argv) {
    // 读取图像
    cv::Mat img1 = cv::imread("images/0.png",IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("images/1.png",IMREAD_GRAYSCALE);

    // 相机内参矩阵 (需要根据实际情况设置)
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 718.8560, 0, 607.1928, 
                                                      0, 718.8560, 185.2157, 
                                                      0, 0, 1);


    if (img1.empty() || img2.empty()) {
        std::cerr << "无法读取图像！" << std::endl;
        return -1;
    }

    featureMatchingAndPoseEstimation(img1, img2, cameraMatrix);

    return 0;
}
