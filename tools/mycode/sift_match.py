import cv2
import numpy as np
from matplotlib import pyplot as plt
def extract_sift_feature(img):
    #创建SIFT检测器
    sift = cv2.xfeatures2d.SIFT_create()
    #提取图像的特征点和描述子信息
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints,descriptors

def draw_match_image(img1,img2):
    #提取图像的SIFT关键点信息
    keypoints1, descriptors1 = extract_sift_feature(img1)
    keypoints2, descriptors2 = extract_sift_feature(img2)

    #创建一个关键点匹配器,采用L1距离
    # 比率测试（对应 COLMAP 的 max_distance=0.7）
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    # 几何验证（对应 COLMAP 的 max_error=4.0）
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        inliers = mask.sum()
        print(f"内点数量: {inliers}/{len(good_matches)}")
    else:
        print("匹配点不足，跳过几何验证")

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #关键点匹配,通过计算两张图像提取的描述子之间的距离
    matches = bf.match(descriptors1, descriptors2)
    #根据描述子之间的距离进行排序
    matches = sorted(matches, key=lambda x: x.distance)
    

