"""
修改了OCR的读取方式，直接读取json文件中的ocr_result
"""
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(__file__)
import Levenshtein
import json
import uuid
import math
from pathlib import Path
from text_similarity import calculate_similarity
from cal_poi import create_poi
import shutil
import pandas as pd
import zhconv 
import config_signboard_entity as config
from create_sign_new import pair_id_to_sign_ids, SIGNDatabase


def is_point_in_bbox(point, bbox):# -> Any:
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def load_and_process_images(images_path,db):
    """
    加载并处理图像数据
    :param images_path: 图像信息文件路径,images.txt
        # Image list with two lines of data per image:
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # POINTS2D[] as (X, Y, POINT3D_ID)
    :param db: 数据库连接对象,将处理后的招牌实体数据存储到数据库中
    return
    :name_id: {image_name: image_id} image_name形如190_front 不包含类型名
    :correspond_struct_idx:{image_name: [index1, index2, ...]},是kp对应的3d index，所以形状一定是和keypoints_for_all对应的
    :keypoints_for_all:{image_name:np.array([[x1, y1], [x2, y2], ...])}
         """
    correspond_struct_idx = {}
    keypoints_for_all = {}
    pose = []
    name_id = {}
    with open(images_path, 'r') as file:
        for index, line in enumerate(file):
            if line.strip().startswith('#'):
                continue
            elements = line.split()
            if index % 2 == 0:
                # 处理图像信息和相机位姿
                image_id = int(elements[0])
                prior_q = np.array([float(element) for element in elements[1:5]])
                prior_t = np.array([float(element) for element in elements[5:8]])
                image_name = elements[-1]
                image_name = image_name.split('.')[0]
                name_id[image_name] = image_id
                key_points = []
            else:
                # 处理keypoints信息
                for i, element in enumerate(elements):
                    if i % 3 == 0:
                        kp = [float(element)]  # 初始化关键点列表
                    elif i % 3 == 1:
                        kp.append(float(element))  # 添加y坐标
                    else:
                        kp.append(int(element))  # 添加关键点索引
                        key_points.append(kp)  # 完整的关键点信息[x, y, index]

                key_points_array = np.array(key_points, dtype=np.float32)
                correspond_struct_idx[image_name] = [kp[2] for kp in key_points]
                keypoints_for_all[image_name] = key_points_array[:, :2]  # 只保留x, y,不保留索引
                # 存储到数据库
                db.add_image(image_id, image_name, correspond_struct_idx[image_name], prior_q, prior_t) #子表 images
                db.add_keypoints(image_name, keypoints_for_all[image_name]) #子表 keypoints
    return name_id, keypoints_for_all, correspond_struct_idx

def process_detection_file(det_file):
    """处理检测文件，返回边界框列表"""
    with open(det_file, 'r') as file:
        return [list(map(float, line.split())) for line in file if line.strip()]

def read_ocr_file(ocr_file):
    """读取OCR文件，返回文本列表"""
    try:
        with open(ocr_file, 'r') as file:
            return [line.strip().split(' ')[0] for line in file ]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"打开文件 {ocr_file} 时发生错误: {e}")
        return []

def add_sign_data_to_db(db, match, correspond_struct_idx, bboxes, shopsign_texts,key_points):
    """添加标识数据到数据库"""
    sign_ids = []
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        point3d_idx = []
        for j, kp in enumerate(key_points):
            if (correspond_struct_idx[j] != -1 and x_min <= kp[1] <= x_max and y_min <= kp[
                0] <= y_max):  # 特征点在shopsign dete范围内并有对应3d点
                point3d_idx.append(correspond_struct_idx[j])
        text = shopsign_texts[i] if shopsign_texts else ""
        sign_id = db.add_sign(x_min, y_min, x_max, y_max, match, point3d_idx, text)
        sign_ids.append(sign_id)
    return sign_ids

def find_corresponding_sign(feature_matches, features_img1, features_img2, data1, data2, error_matching_dir):

    """
    查找对应的标牌
    :param feature_matches: (m,2) 所有匹配点idx
    :param features_img1, features_img2:(n1,2),(n2,2) 所有特征点坐标
    :param data1, data2: [sign]

    return
    :corresponding_ids: [[sign_id1, sign_id2, match_score]]
    :corresponding_idx: [np.array([[feature_idx1, feature_idx2]])]   
    """
    args = config.parse_args()
    lambda1 = args.lambda1
    threshold = args.threshold
    k = args.k
    corresponding_idx = []
    corresponding_ids = []
    
    # 确保目录存在
    os.makedirs(error_matching_dir, exist_ok=True)
    output_file = os.path.join(error_matching_dir, "匹配错误.txt")
    
    #如果任一图片的特征点为空，直接返回空结果
    if features_img1 is None or features_img2 is None:
        return corresponding_ids,corresponding_idx
    else:
        for sign1 in data1:
            # 使用sign_id作为键
            match_counts = {sign2['sign_id']: 0 for sign2 in data2} #{sign_id: 0}
            match_idxs = {sign2['sign_id']: [] for sign2 in data2} #{sign_id: []}
            match_scores = {sign2['sign_id']: 0 for sign2 in data2} #{sign_id: 0}
            
            # 存储文本相似度和特征点匹配率
            text_similarities = {sign2['sign_id']: 0 for sign2 in data2}
            feature_match_ratios = {sign2['sign_id']: 0 for sign2 in data2}
            match_points_counts = {sign2['sign_id']: 0 for sign2 in data2}
            total_points_counts = {sign2['sign_id']: 0 for sign2 in data2}
        
            # 计算在bbox1内的所有特征点数量 all_points_in_bbox1 
            bbox1 = sign1["board_contour"]
            all_points_in_bbox1 = 0
            for i in range(len(features_img1)):
                feature_point = features_img1[i]
                if is_point_in_bbox(feature_point, bbox1):
                    all_points_in_bbox1 += 1
            
            # 计算在bbox1内且有匹配关系的特征点
            points_in_bbox1_matched = []
            for feature_idx1, feature_idx2 in feature_matches:
                feature_point_img1 = features_img1[feature_idx1]
                if is_point_in_bbox(feature_point_img1, bbox1):
                    points_in_bbox1_matched.append((feature_idx1, feature_idx2))
            
            # 提取sign1的文本
            text1 = ''
            has_text1 = False
            if sign1.get("ocr_result") and sign1["ocr_result"]:
                text1 = ' '.join(sign1["ocr_result"])
                text1 = zhconv.convert(text1, 'zh-hans')
                has_text1 = True

            for sign2 in data2:
                bbox2 = sign2["board_contour"]
                # 计算在bbox2内的所有特征点
                all_points_in_bbox2 = 0
                for i in range(len(features_img2)):
                    feature_point = features_img2[i]
                    if is_point_in_bbox(feature_point, bbox2):
                        all_points_in_bbox2 += 1
                # 提取sign2的文本
                text2 = ''
                text2 = zhconv.convert(text2, 'zh-hans')
                has_text2 = False
                if sign2.get("ocr_result") and sign2["ocr_result"]:
                    text2 = ' '.join(sign2["ocr_result"])
                    has_text2 = True

                # 计算匹配的特征点
                for feature_idx1, feature_idx2 in points_in_bbox1_matched:
                    feature_point_img2 = features_img2[feature_idx2]
                    if is_point_in_bbox(feature_point_img2, bbox2):
                        match_counts[sign2['sign_id']] += 1
                        match_idxs[sign2['sign_id']].append([feature_idx1, feature_idx2])
                match_points_counts[sign2['sign_id']] = match_counts[sign2['sign_id']]
                total_points_counts[sign2['sign_id']] = max(all_points_in_bbox1, all_points_in_bbox2)

                # 计算特征点匹配率 - 使用两个招牌中所有特征点的最大值作为分母
                max_points = max(all_points_in_bbox1, all_points_in_bbox2)
                feature_match_ratio = match_counts[sign2['sign_id']] / max_points if max_points > 3 else 0
                feature_match_ratios[sign2['sign_id']] = feature_match_ratio
                
                # 计算文本相似度    
                if has_text1 and has_text2:
                    levenshtein_distance = Levenshtein.distance(text1, text2)
                    normalized_levenshtein_distance = levenshtein_distance / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 1
                    text_similarity_score = 1 - normalized_levenshtein_distance
                    text_similarities[sign2['sign_id']] = text_similarity_score
                else:
                    text_similarity_score = 0
                match_scores[sign2['sign_id']] = lambda1 * text_similarity_score + (1-lambda1) * feature_match_ratio

            # 找出最佳匹配
            max_match_score = max(match_scores.values()) if match_scores else 0
            if max_match_score > threshold:
                corresponding_id = max(match_scores, key=match_scores.get)
                corresponding_sign = next((sign2 for sign2 in data2 if sign2['sign_id'] == corresponding_id), None)
                
                if corresponding_sign:
                    # 获取匹配信息
                    text_similarity = text_similarities.get(corresponding_id, 0)
                    feature_ratio = feature_match_ratios.get(corresponding_id, 0)
                    match_points = match_points_counts.get(corresponding_id, 0)
                    total_points = total_points_counts.get(corresponding_id, 0)
                    
                    # 添加匹配分数到返回结果
                    corresponding_ids.append([sign1['sign_id'], corresponding_id, max_match_score])
                    corresponding_idx.append(np.array(match_idxs[corresponding_id]))
            
        return corresponding_ids, corresponding_idx

def find_and_store_matches(db, match_idx, keypoints1, keypoints2, bboxes_img1, bboxes_img2, sign_ids1, sign_ids2):
    """查找匹配的检测框并存储到数据库"""
    corresponding_bbox, corresponding_idx = find_corresponding_sign(match_idx, keypoints1, keypoints2, bboxes_img1, bboxes_img2)
    for i, bboxes in enumerate(corresponding_bbox):
        bbox1, bbox2 = bboxes
        sign_id1 = sign_ids1[bboxes_img1.index(bbox1)]
        sign_id2 = sign_ids2[bboxes_img2.index(bbox2)]
        db.add_matches(sign_id1, sign_id2, corresponding_idx[i])


def load_points3d(points3D_path,db):
    points3d = {}
    with open(points3D_path, 'r') as file:
        for line in file:
            # 分割每一行的字符串，转换为元素列表
            elements = line.split()
            if elements[0] == '#':
                continue
            # 提取前四个元素并将它们转换为浮点数（假设前四个都是可以转换为浮点数的）
            elements = [float(element) for element in elements[:4]]
            key = int(elements[0])  # 将第一个元素转换为整数键
            values = np.array(elements[1:4], dtype=np.float32)  # 将后三个元素转换为浮点数的np.array
            points3d[key] = values
    db.add_points3d(points3d)
    return points3d

def load_sign(dect_path, relevant_images, keypoints_for_all, correspond_struct_idx, db):
    """
    从dect_path下的JSON文件加载sign数据到数据库中
    
    参数:
        dect_path: 检测结果JSON文件所在目录路径
        relevant_images: 相关图片列表
        keypoints_for_all: 所有图片的关键点
        correspond_struct_idx: 关键点对应的3D点索引
        db: 数据库对象
    """
    import os
    import json
    
    for image_name in relevant_images:
        json_path = os.path.join(dect_path, f"{image_name}.json")
        if not os.path.exists(json_path):
            continue
        
        # 读取当前图片的sign数据
        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                signs_data = json.load(json_file)
        except Exception as e:
            print(f"读取JSON文件出错 {json_path}: {e}")
            continue
        
        # 处理该图片中的每个sign
        for sign_id, sign_info in signs_data.items():
            # 确保sign_id是字符串格式
            sign_id = str(sign_id)
            
            # 获取边界框
            board_contour = sign_info.get("board_contour", [0, 0, 0, 0])
            x_min, y_min, x_max, y_max = board_contour
            
            # 获取OCR结果
            ocr_result = sign_info.get("ocr_result", [])
            ocr_result = [zhconv.convert(text, 'zh-hans') for text in ocr_result]
            text_store = json.dumps(ocr_result, ensure_ascii=False)
            
            # 获取置信度
            confidence = sign_info.get("confidence", 1.0)
            
            # 找到对应的3D点
            point3d_idx = []
            if image_name in keypoints_for_all and image_name in correspond_struct_idx:
                key_points = keypoints_for_all[image_name]
                for i, kp in enumerate(key_points):
                    if (correspond_struct_idx[image_name][i] != -1 and 
                        x_min <= kp[1] <= x_max and 
                        y_min <= kp[0] <= y_max):
                        point3d_idx.append(correspond_struct_idx[image_name][i])
            
            # 添加到数据库
            db.add_sign(
                sign_id=sign_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                correspond_points3d_idx=point3d_idx,
                text_content=text_store,
                text_confidence=confidence,
                image_name=image_name)

def save_poi(POI_list,output_path,group_id = None):
    """
    POI_list = [{
    {'text_content': list[str], 
     'coordinate': [lat, lon, alt], 
     'sign_id': list[sign_id], 
     'top_k_sign_id': list[sign_id]
    }]
    保存的时候加了一个group_id字段
    data[poi_id] = item
    """
    data = {}
    poi_id =0
    for item in POI_list:
        # if all(math.isnan(coord) for coord in item['coordinate']):
        #     continue
        if group_id:
            item['group_id'] = int(group_id)
        data[poi_id] = item
        poi_id += 1
    with open(output_path,'w',encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def get_relevant_images(group_dir, group_gps_csv, colmap_images_path):
    """
    从CSV文件中获取每个组相关的图片列表并与COLMAP重建的图片列表取交集
    
    参数:
        group_dir: 组文件夹路径
        group_gps_csv: CSV文件名，包含图片信息
        colmap_images_path: COLMAP重建的images.txt文件路径
    
    返回:
        列表，包含相关的图片名称
    """
    import pandas as pd
    import os
    
    # 读取CSV文件中的图片列表
    csv_path = os.path.join(group_dir, group_gps_csv)
    if not os.path.exists(csv_path):
        return []
    
    csv_data = pd.read_csv(csv_path)
    csv_images = set()
    for _, row in csv_data.iterrows():
        image_name = row['name']
        # 确保图片名称格式一致（移除扩展名用于比较）
        if '.' in image_name:
            image_name = image_name.rsplit('.', 1)[0]
        csv_images.add(image_name)
    
    # 读取COLMAP重建的图片列表
    colmap_images = set()
    with open(colmap_images_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                elements = line.split()
                if len(elements) > 9:  # 确保行有足够的元素
                    image_name = elements[-1]
                    if '.' in image_name:
                        image_name = image_name.rsplit('.', 1)[0]
                    colmap_images.add(image_name)
    
    # 返回两个列表的交集
    return list(csv_images.intersection(colmap_images))


def create_signs(dect_path, output_dir, colmap_dir, group_dir, group_gps_csv, sign_path='sign.db', error_matching_dir=None):
    """
    从检测结果创建sign并处理匹配
    
    参数:
        dect_path: 检测结果JSON文件所在目录路径
        output_dir: 输出目录
        colmap_dir: COLMAP重建目录
        group_dir: 组文件夹路径
        group_gps_csv: CSV文件名，包含图片信息
        sign_path: 输出的签名数据库路径
        error_matching_dir: 错误匹配输出目录
    """
    # 连接数据库
    images_path = os.path.join(colmap_dir, 'images.txt')
    matches_file = os.path.join(output_dir, 'matches.txt')
    points3D_path = os.path.join(colmap_dir,'points3D.txt')
    
    if os.path.exists(sign_path):
        os.remove(sign_path)
    if not os.path.exists(points3D_path):
        return
    
    # 获取相关的图片列表
    relevant_images = get_relevant_images(group_dir, group_gps_csv, images_path)
    if not relevant_images:
        print(f"No relevant images found in group: {os.path.basename(group_dir)}. Skipping sign creation.")
        return

    db = SIGNDatabase.connect(sign_path)
    db.create_tables()

    name_id, keypoints_for_all, correspond_struct_idx = load_and_process_images(images_path, db)
    points3d = load_points3d(points3D_path, db)

    # 加载sign数据
    load_sign(dect_path, relevant_images, keypoints_for_all, correspond_struct_idx, db)

    # 处理匹配
    with open(matches_file, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            if len(parts) < 2:
                continue
                
            matches = parts[0].split(' ')
            
            # 去掉文件扩展名进行比较
            match_images = [image_name.split('.')[0] for image_name in matches]
            if parts[1] == '' or not all(image in relevant_images for image in match_images):
                continue
                
            # 处理匹配数据
            match_idx = list(map(int, parts[1].split(' ')))
            match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
            
            # 获取匹配图片的关键点
            keypoints1 = keypoints_for_all.get(match_images[0])
            keypoints2 = keypoints_for_all.get(match_images[1])
            
            # 获取每张图片的标志
            json_path1 = os.path.join(dect_path, f"{match_images[0]}.json")
            json_path2 = os.path.join(dect_path, f"{match_images[1]}.json")
            
            data1 = []
            data2 = []
            
            # 读取第一张图片的标志数据
            if os.path.exists(json_path1):
                with open(json_path1, 'r', encoding='utf-8') as f:
                    sign_data = json.load(f)
                    for sign_id, sign_info in sign_data.items():
                        sign_info['sign_id'] = sign_id
                        sign_info['name'] = match_images[0]  # 添加图片名称字段，用于错误匹配报告
                        data1.append(sign_info)
            
            # 读取第二张图片的标志数据
            if os.path.exists(json_path2):
                with open(json_path2, 'r', encoding='utf-8') as f:
                    sign_data = json.load(f)
                    for sign_id, sign_info in sign_data.items():
                        sign_info['sign_id'] = sign_id
                        sign_info['name'] = match_images[1]
                        data2.append(sign_info)
            
            # 处理匹配
            if data1 and data2:
                corresponding_ids, corresponding_idx = find_corresponding_sign(
                    match_idx, keypoints1, keypoints2, data1, data2, error_matching_dir
                ) 
                
                if corresponding_ids:
                    for i, corresponding_id in enumerate(corresponding_ids):
                        try:
                            # 检查数据库的add_matches方法支持的参数
                            db.add_matches(corresponding_id[0], corresponding_id[1], corresponding_idx[i], corresponding_id[2])
                        except TypeError:
                            # 如果方法不支持match_score参数，则使用旧的调用方式，不add score
                            db.add_matches(corresponding_id[0], corresponding_id[1], corresponding_idx[i])
    
    db.commit()
    db.close()

# 主程序中的调用保持不变
if __name__ == "__main__":
    #路径设置
    args = config.parse_args()
    group_path = args.group_path
    groups = os.listdir(group_path)
    dect_path = args.dect_path
    group_gps_csv = args.group_gps_csv 
    poi_path = args.poi_json

    for i in groups:
        print(i)
        group_dir = os.path.join(group_path, i)
        poi_json = os.path.join(group_dir, poi_path)
        colmap_dir = os.path.join(group_dir, 'colmap/aligned')
        
        # 每个组的检测结果路径
        group_dect_path = os.path.join(dect_path, i) if os.path.exists(os.path.join(dect_path, i)) else dect_path
        sign_db = os.path.join(group_dir, args.sign_path)  
        #创建和对齐招牌实例
        create_signs(
            group_dect_path, 
            group_dir, 
            colmap_dir, 
            group_dir, 
            group_gps_csv, 
            sign_db
        )
        
        #创建和保存POI
        if os.path.exists(sign_db):
            POIS = create_poi(sign_db, args.k, match_threshold=args.match_threshold)
            save_poi(POIS, poi_json, i)