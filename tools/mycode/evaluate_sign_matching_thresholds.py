import os
import json
import sqlite3
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt
import Levenshtein
import config.config_signboard_entity_area3 as config
from collections import defaultdict
from create_sign import SIGNDatabase
import shutil
lambda1 = 0.6
current_dir = os.path.dirname(__file__)

def find_sign_db_files(root_directory, sign_base_path, mode):
    """Find all sign db files for a specific mode in the group directories"""
    sign_db_path = sign_base_path.replace('.db', f'_{mode}.db')
    sign_db_paths = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if sign_db_path in filenames:
            sign_db_paths.append(os.path.join(dirpath, sign_db_path))
    return sign_db_paths

def load_ground_truth(json_path):
    """Load ground truth from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 构建真实聚类
    poi_groups = {} #{poi_id:set(sign_group)}
    sign_to_poi = {} #{sign_id:poi_id},用于快速查找sign属于哪个poi
    
    for poi in data['poi']:
        poi_id = poi['poi_id']
        sign_group = poi['sign_group']
        
        poi_groups[poi_id] = set(sign_group)
        
        # 记录每个sign_id属于哪个POI
        for sign_id in sign_group:
            sign_to_poi[sign_id] = poi_id
    
    return poi_groups, sign_to_poi

def is_point_in_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def find_corresponding_sign_with_scores(feature_matches, features_img1, features_img2, data1, data2, match_mode='hybrid'):
    """
    查找对应的标牌并返回匹配分数，不再根据阈值过滤
    :return: 匹配列表和分数 [{'sign_id1': id1, 'sign_id2': id2, 'score': score}]
    """
    match_results = []
    
    if features_img1 is None or features_img2 is None:
        return match_results
    
    for sign1 in data1:
        # 使用sign_id作为键
        match_counts = {sign2['sign_id']: 0 for sign2 in data2}
        match_idxs = {sign2['sign_id']: [] for sign2 in data2}
        match_scores = {sign2['sign_id']: 0 for sign2 in data2}
        
        # 存储文本相似度和特征点匹配率
        text_similarities = {sign2['sign_id']: 0 for sign2 in data2}
        feature_match_ratios = {sign2['sign_id']: 0 for sign2 in data2}
        
        # 先计算在bbox1内的特征点
        bbox1 = sign1["board_contour"]
        points_in_bbox1 = []
        for feature_idx1, feature_idx2 in feature_matches:
            feature_point_img1 = features_img1[feature_idx1]
            if is_point_in_bbox(feature_point_img1, bbox1):
                points_in_bbox1.append((feature_idx1, feature_idx2))
        count0 = len(points_in_bbox1)

        # 提取sign1的文本
        text1 = ''
        has_text1 = False
        if sign1.get("ocr_result") and any(text_entry.get('text') for text_entry in sign1["ocr_result"]):
            text1 = ' '.join([text_entry.get('text', '') for text_entry in sign1["ocr_result"]])
            has_text1 = True

        for sign2 in data2:
            bbox2 = sign2["board_contour"]
            
            # 提取sign2的文本
            text2 = ''
            has_text2 = False
            if sign2.get("ocr_result") and any(text_entry.get('text') for text_entry in sign2["ocr_result"]):
                text2 = ' '.join([text_entry.get('text', '') for text_entry in sign2["ocr_result"]])
                has_text2 = True
            
            # 计算特征点匹配
            for feature_idx1, feature_idx2 in points_in_bbox1:
                feature_point_img2 = features_img2[feature_idx2]
                if is_point_in_bbox(feature_point_img2, bbox2):
                    match_counts[sign2['sign_id']] += 1
                    match_idxs[sign2['sign_id']].append([feature_idx1, feature_idx2])
            
            # 计算特征点匹配率
            feature_match_ratio = match_counts[sign2['sign_id']] / count0 if count0 > 3 else 0
            feature_match_ratios[sign2['sign_id']] = feature_match_ratio
            
            # 计算文本相似度
            if has_text1 and has_text2:
                levenshtein_distance = Levenshtein.distance(text1, text2)
                normalized_levenshtein_distance = levenshtein_distance / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 1
                text_similarity_score = 1 - normalized_levenshtein_distance
                text_similarities[sign2['sign_id']] = text_similarity_score
            
            # 根据不同匹配模式计算最终分数
            if match_mode == 'text_only':
                # 仅使用文本相似度
                if has_text1 and has_text2:
                    match_scores[sign2['sign_id']] = text_similarity_score
                else:
                    match_scores[sign2['sign_id']] = 0
            
            elif match_mode == 'feature_only':
                # 仅使用特征点匹配
                if count0 > 3:
                    match_scores[sign2['sign_id']] = feature_match_ratio
                else:
                    match_scores[sign2['sign_id']] = 0
            
            else:  # hybrid模式(原始逻辑)
                
                if has_text1 and has_text2:
                    if count0 > 3:
                        # 综合文本和特征点
                        match_scores[sign2['sign_id']] = lambda1 * text_similarity_score + (1-lambda1) * feature_match_ratio
                    else:
                        # 仅文本
                        match_scores[sign2['sign_id']] = lambda1 * text_similarity_score
                else:
                    # 至少一方没有文本，仅依赖特征点匹配，退化到纯特征点匹配时要把阈值调高
                    if count0 > 3:
                        #match_scores[sign2['sign_id']] = 0.5 * feature_match_ratio
                        match_scores[sign2['sign_id']] = feature_match_ratio
                    else:
                        match_scores[sign2['sign_id']] = 0
        
        # 找出所有有分数的匹配，不再根据阈值过滤
        for sign_id2, score in match_scores.items():
            if score > 0:  # 只保存有分数的匹配
                match_results.append({
                    'sign_id1': sign1['sign_id'],
                    'sign_id2': sign_id2,
                    'score': score
                })
    
    return match_results

def create_signs_with_all_matches(json_path, output_dir, colmap_dir, results_json, match_mode='hybrid'):
    """创建标牌匹配，保存所有匹配结果及其分数到JSON文件中"""
    images_path = os.path.join(colmap_dir, 'images.txt')
    matches_file = os.path.join(output_dir, 'matches.txt')
    
    # 检查必要文件是否存在
    if not all(os.path.exists(f) for f in [images_path, matches_file]):
        print(f"  缺少必要文件，跳过处理 {output_dir}")
        return None

    # 获取相关的图片列表
    relevant_images = get_relevant_images(json_path, images_path)
    if not relevant_images:
        print(f"  未找到相关图片，跳过处理 {output_dir}")
        return None

    # 加载图像和关键点信息
    keypoints_for_all, correspond_struct_idx = load_image_keypoints(images_path)
    
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        sign_data = [sign for sign in data['sign'] if sign['image_name'] in relevant_images]

    # 存储所有匹配结果
    all_matches = []
    
    # 处理匹配
    with open(matches_file, 'r',  encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            if len(parts) < 2:
                continue
                
            matches = parts[0].split(' ')
            if parts[1] == '' or not all(match in relevant_images for match in matches):
                continue
                
            matches = [image_name.split('.')[0] for image_name in matches]
            try:
                match_idx = list(map(int, parts[1].split(' ')))
                match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
            except (ValueError, IndexError):
                print(f"  跳过无效匹配行: {line}")
                continue
            
            keypoints1 = keypoints_for_all.get(matches[0])
            keypoints2 = keypoints_for_all.get(matches[1])
            
            data1 = [item for item in sign_data if item['image_name'].split('.')[0] == matches[0]]
            data2 = [item for item in sign_data if item['image_name'].split('.')[0] == matches[1]]
            
            if data1 and data2:
                match_results = find_corresponding_sign_with_scores(
                    match_idx, keypoints1, keypoints2, data1, data2, match_mode)
                all_matches.extend(match_results)
    
    # 为每个match添加group标识
    group_id = os.path.basename(output_dir)
    for match in all_matches:
        match['group'] = group_id
    
    return all_matches

def collect_all_group_matches(mode, groups, group_path, json_path):
    """收集所有组的匹配结果"""
    all_matches = []
    valid_groups = []
    
    print(f"\n为 {mode} 模式收集所有组的匹配结果...")
    
    for i in groups:
        print(f"  处理组 {i}")
        group_dir = os.path.join(group_path, i)
        colmap_dir = os.path.join(group_dir, 'colmap/aligned')
        
        if not os.path.exists(os.path.join(colmap_dir, 'points3D.txt')):
            print(f"  组 {i} 缺少colmap数据，跳过")
            continue
            
        matches = create_signs_with_all_matches(json_path, group_dir, colmap_dir, None, mode)
        if matches:
            print(f"  收集了 {len(matches)} 个匹配")
            all_matches.extend(matches)
            valid_groups.append(i)
        else:
            print(f"  组 {i} 无有效匹配，跳过")
    
    print(f"共收集了 {len(all_matches)} 个匹配，来自 {len(valid_groups)} 个有效组")
    return all_matches

def get_relevant_images(json_path, images_path):
    """获取JSON和COLMAP中都存在的图片"""
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        json_images = set(sign['image_name'] for sign in data['sign'])

    colmap_images = set()
    with open(images_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                elements = line.split()
                if len(elements) > 9:
                    image_name = elements[-1]
                    colmap_images.add(image_name)

    return list(json_images.intersection(colmap_images))

def load_image_keypoints(images_path):
    """从COLMAP images.txt加载图像关键点信息"""
    keypoints_for_all = {}
    correspond_struct_idx = {}
    
    with open(images_path, 'r') as file:
        for index, line in enumerate(file):
            if line.strip().startswith('#'):
                continue
                
            elements = line.split()
            if index % 2 == 0:
                image_name = elements[-1].split('.')[0]
                key_points = []
            else:
                for i, element in enumerate(elements):
                    if i % 3 == 0:
                        kp = [float(element)]
                    elif i % 3 == 1:
                        kp.append(float(element))
                    else:
                        kp.append(int(element))
                        key_points.append(kp)

                key_points_array = np.array(key_points, dtype=np.float32)
                correspond_struct_idx[image_name] = [kp[2] for kp in key_points]
                keypoints_for_all[image_name] = key_points_array[:, :2]
                
    return keypoints_for_all, correspond_struct_idx

def build_predicted_clusters_with_threshold(matches, threshold):
    """根据阈值构建预测聚类"""
    G = nx.Graph()
    
    # 只添加分数大于阈值的匹配作为边
    for match in matches:
        if match['score'] >= threshold:
            G.add_edge(match['sign_id1'], match['sign_id2'])
    
    # 提取连通分量作为聚类
    predicted_clusters = []
    for component in nx.connected_components(G):
        predicted_clusters.append(set(component))
    
    return predicted_clusters

def calculate_pairwise_metrics(poi_groups, predicted_clusters):
    """计算成对指标"""
    # 构建真实的同类pairs
    true_pairs = set()
    for cluster in poi_groups.values():
        for sign1 in cluster:
            for sign2 in cluster:
                if sign1 < sign2:  # 避免重复
                    true_pairs.add((sign1, sign2))
    
    # 构建预测的同类pairs
    pred_pairs = set()
    for cluster in predicted_clusters:
        for sign1 in cluster:
            for sign2 in cluster:
                if sign1 < sign2:  # 避免重复
                    pred_pairs.add((sign1, sign2))
    
    # 计算指标
    true_positives = len(true_pairs.intersection(pred_pairs))
    precision = true_positives / len(pred_pairs) if pred_pairs else 0
    recall = true_positives / len(true_pairs) if true_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_clustering_metrics(poi_groups, predicted_clusters, all_signs, sign_to_poi):
    """计算聚类比较指标"""
    # 将聚类转换为列表形式，适用于scikit-learn指标
    true_labels = []
    pred_labels = []
    
    true_cluster_map = sign_to_poi  # {sign_id: poi_id}
    
    pred_cluster_map = {}  # {sign_id: pred_cluster_id}
    for i, cluster in enumerate(predicted_clusters):
        for sign_id in cluster:
            pred_cluster_map[sign_id] = i
    max_cluster_id = max(pred_cluster_map.values()) if pred_cluster_map else -1
    singleton_id = max_cluster_id + 1

    # 为所有sign分配标签
    for sign_id in all_signs:
        true_labels.append(true_cluster_map.get(sign_id, -1))  # -1表示未分类
        if sign_id not in pred_cluster_map:
            pred_labels.append(singleton_id)
            singleton_id += 1
        else:
            pred_labels.append(pred_cluster_map.get(sign_id))

    # 计算调整兰德指数和调整互信息
    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    
    return {
        'adjusted_rand_index': ari,
        'adjusted_mutual_info': ami
    }

def evaluate_threshold(matches, poi_groups, sign_to_poi, all_signs, threshold):
    """评估特定阈值下的匹配性能"""
    predicted_clusters = build_predicted_clusters_with_threshold(matches, threshold)
    
    # 计算指标
    pairwise_metrics = calculate_pairwise_metrics(poi_groups, predicted_clusters)
    clustering_metrics = calculate_clustering_metrics(poi_groups, predicted_clusters, all_signs, sign_to_poi)
    
    return {
        'threshold': threshold,
        'precision': pairwise_metrics['precision'],
        'recall': pairwise_metrics['recall'],
        'f1': pairwise_metrics['f1'],
        'adjusted_rand_index': clustering_metrics['adjusted_rand_index'],
        'adjusted_mutual_info': clustering_metrics['adjusted_mutual_info']
    }

def get_all_signs(poi_groups, matches):
    """获取所有sign"""
    all_signs = set()
    for cluster in poi_groups.values():
        all_signs.update(cluster)
    
    for match in matches:
        all_signs.add(match['sign_id1'])
        all_signs.add(match['sign_id2'])
    
    return all_signs

def evaluate_mode_with_thresholds(mode, groups, group_path, json_path, thresholds, output_dir):
    """评估特定模式在多个阈值下的性能"""
    # 收集所有组的匹配结果
    all_matches = collect_all_group_matches(mode, groups, group_path, json_path)
    
    # 保存综合结果到JSON
    results_json = os.path.join(output_dir, f'all_matches_{mode}.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=4)
    
    # 如果没有匹配结果，返回空列表
    if not all_matches:
        #print(f"没有找到任何匹配结果，{mode} 模式评估跳过")
        return []
    
    # 加载真实值
    poi_groups, sign_to_poi = load_ground_truth(json_path)
    all_signs = get_all_signs(poi_groups, all_matches)
    
    print(f"共加载了 {len(all_matches)} 个匹配和 {len(poi_groups)} 个POI组")
    
    # 评估不同阈值
    threshold_results = []
    for threshold in thresholds:
        print(f"评估阈值 {threshold:.2f}...")
        result = evaluate_threshold(all_matches, poi_groups, sign_to_poi, all_signs, threshold)
        threshold_results.append(result)
        print(f"  F1: {result['f1']:.4f}, ARI: {result['adjusted_rand_index']:.4f}")
    
    return threshold_results

def visualize_threshold_results(results_dict, output_dir):
    """可视化不同阈值和模式的性能对比"""
    # 转换为DataFrame以便绘图
    df_list = []
    for mode, results in results_dict.items():
        df = pd.DataFrame(results)
        df['mode'] = mode
        df_list.append(df)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # 创建可视化目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制F1分数随阈值变化的曲线
    plt.figure(figsize=(10, 6))
    for mode in results_dict.keys():
        mode_df = df_combined[df_combined['mode'] == mode]
        plt.plot(mode_df['threshold'], mode_df['f1'], marker='o', label=f'{mode}')
    
    plt.xlabel('阈值')
    plt.ylabel('F1分数')
    plt.title('不同模式和阈值的F1分数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'threshold_f1.png'), dpi=300)
    
    # 绘制调整兰德指数随阈值变化的曲线
    plt.figure(figsize=(10, 6))
    for mode in results_dict.keys():
        mode_df = df_combined[df_combined['mode'] == mode]
        plt.plot(mode_df['threshold'], mode_df['adjusted_rand_index'], marker='o', label=f'{mode}')
    
    plt.xlabel('阈值')
    plt.ylabel('调整兰德指数')
    plt.title('不同模式和阈值的调整兰德指数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'threshold_ari.png'), dpi=300)
    
    # 输出最佳阈值
    best_thresholds = {}
    for mode in results_dict.keys():
        mode_df = df_combined[df_combined['mode'] == mode]
        best_f1_idx = mode_df['f1'].idxmax()
        best_threshold = mode_df.loc[best_f1_idx, 'threshold']
        best_f1 = mode_df.loc[best_f1_idx, 'f1']
        best_thresholds[mode] = {
            'threshold': best_threshold,
            'f1': best_f1
        }
    
    # 保存最佳阈值到CSV
    best_df = pd.DataFrame([
        {
            'mode': mode,
            'best_threshold': data['threshold'],
            'best_f1': data['f1']
        } for mode, data in best_thresholds.items()
    ])
    best_df.to_csv(os.path.join(output_dir, 'best_thresholds.csv'), index=False)
    
    # 显示结果
    print("\n最佳阈值:")
    for mode, data in best_thresholds.items():
        print(f"{mode}: 阈值 = {data['threshold']:.2f}, F1 = {data['f1']:.4f}")
    
    # 创建性能指标表格
    pivot_df = df_combined.pivot_table(
        index=['mode', 'threshold'],
        values=['precision', 'recall', 'f1', 'adjusted_rand_index', 'adjusted_mutual_info']
    ).round(4)
    
    pivot_df.to_csv(os.path.join(output_dir, 'threshold_metrics.csv'))
    print(f"\n详细结果已保存至 {output_dir} 目录")

if __name__ == "__main__":
    args = config.parse_args()
    group_path = args.group_path
    groups = os.listdir(group_path)
    json_path = args.shopsign_json
    
    # 定义要测试的匹配模式和阈值范围
    match_modes = ['hybrid', 'text_only', 'feature_only']
    thresholds = thresholds = np.arange(0.15, 0.56, 0.01).tolist()  # 从0.15到0.55，间隔为0.01
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, 'threshold_evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 结果字典
    results_dict = {}
    
    # 对每种匹配模式运行评估
    for mode in match_modes:
        print(f"\n评估 {mode} 匹配模式在不同阈值下的性能...")
        threshold_results = evaluate_mode_with_thresholds(mode, groups, group_path, json_path, thresholds, output_dir)
        if threshold_results:
            results_dict[mode] = threshold_results
    
    # 可视化并保存结果
    if results_dict:
        visualize_threshold_results(results_dict, output_dir)
        print("\n所有组的汇总结果已保存")
    else:
        print("没有找到任何有效的匹配结果，无法生成报告")
