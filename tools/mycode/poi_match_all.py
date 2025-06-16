from gensim import corpora
from gensim import models
from gensim import similarities
import jieba
import csv
import os
from geocoding import gcj02_to_wgs84,wgs84_to_gcj02
from ecef2gps import gps2ecef
import numpy as np
from zhconv import convert
import config
import json
from recall_type import read_json,filter_poi
import Levenshtein
import math
from haversine import haversine, Unit
import config_signboard_entity as config
# 添加到文件顶部的导入部分
import matplotlib.pyplot as plt
import seaborn as sns
import os

spatial_threshold = 100
sim_name_threshold = 0.5

#from corpora.corpus import Corpus
# 1 分词
# 1.1 历史比较文档的分词
#距离阈值
max_distance = 100
#相似度阈值
min_similarity =0.6

def plot_distance_boxplot(output_path, save_dir=None):
    """
    绘制匹配POI的距离分布箱型图（横向显示）
    
    Args:
        output_path: 匹配结果文件路径
        save_dir: 图表保存目录，默认保存到结果文件相同目录
    """
    # 读取匹配结果
    with open(output_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref_pois = ref_data.get("poi", [])
    
    # 提取所有距离值
    distances = []
    for poi in ref_pois:
        if "match_POI" in poi and "distance" in poi["match_POI"]:
            distances.append(poi["match_POI"]["distance"])
    
    # 检查是否有距离数据
    if not distances:
        print("No distance data available for plotting")
        return
    
    # 设置绘图风格
    plt.figure(figsize=(10, 6))
    
    # 创建横向箱型图
    box = plt.boxplot(
        distances,
        vert=False,  # 横向显示
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10"
        }
    )
    
    # 样式设置 - 蓝色箱体
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置图表标题和标签
    plt.title('Distribution of POI Matching Distances', fontsize=16)
    plt.xlabel('Distance Error (m)', fontsize=14)
    plt.ylabel('', fontsize=14)  # 横向箱型图不需要y轴标签
    
    # 添加统计信息（英文）
    stats_text = (
        f"Mean: {np.mean(distances):.2f} m\n"
        f"Median: {np.median(distances):.2f} m\n"
        f"Min: {min(distances):.2f} m\n"
        f"Max: {max(distances):.2f} m\n"
        f"Count: {len(distances)}"
    )
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 设置保存路径
    if save_dir is None:
        save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'poi_distance_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Distance distribution plot saved to {save_path}")
    
    # 关闭图形
    plt.close()
    
def process_grouped_pois(output_path, groups_path):
    """处理分组POI，对于同一组POI，只要有一个匹配上了，就删除组中没有匹配上的POI。
    如果一个组中没有任何POI匹配成功，则只保留该组中的一个POI"""
    # 读取匹配结果
    with open(output_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref_pois = ref_data.get("poi", [])
    
    # 读取分组信息
    with open(groups_path, 'r', encoding='utf-8') as f:
        groups_data = json.load(f)
        groups = groups_data.get("groups", [])
    
    # 创建POI ID到索引的映射，方便快速查找
    poi_indices = {poi.get("poi_id"): i for i, poi in enumerate(ref_pois) if "poi_id" in poi}
    
    # 处理每个组
    pois_to_remove = []  # 存储需要删除的POI索引
    
    for group in groups:
        poi_ids = group.get("poi_ids", [])
        
        # 找出组内已匹配的POI和未匹配的POI
        matched_pois = []
        unmatched_pois = []
        group_poi_indices = []  # 存储组内所有POI的索引
        
        for poi_id in poi_ids:
            if poi_id in poi_indices:
                poi_idx = poi_indices[poi_id]
                group_poi_indices.append(poi_idx)
                poi = ref_pois[poi_idx]
                if "match_POI" in poi:
                    matched_pois.append(poi_idx)
                else:
                    unmatched_pois.append(poi_idx)
        
        # 场景1：如果组内有POI匹配上了，删除未匹配的POI
        if matched_pois:
            pois_to_remove.extend(unmatched_pois)
        
        # 场景2：如果组内没有POI匹配上且组内有多个POI，保留一个POI，删除其他
        elif len(unmatched_pois) > 1:
            # 保留第一个POI，删除其他
            pois_to_remove.extend(unmatched_pois[1:])
    
    # 从后向前删除POI，避免索引变化问题
    for idx in sorted(pois_to_remove, reverse=True):
        ref_pois.pop(idx)
    
    # 保存修改后的结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ref_data, f, ensure_ascii=False, indent=4)

def split_chinese_english(text):
    """
    将混合的中英文名称拆分为中文部分和英文部分
    """
    # 判断字符是否为中文
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'
    
    # 判断字符是否为英文字母或数字
    def is_english(char):
        return 'a' <= char.lower() <= 'z' or '0' <= char <= '9'
    
    # 按空格分割
    parts = text.split()
    chinese_parts = []
    english_parts = []
    
    # 根据第一个字符判断每部分是中文还是英文
    for part in parts:
        if part and is_chinese(part[0]):
            chinese_parts.append(part)
        elif part and is_english(part[0]):
            english_parts.append(part)
        else:
            # 对于无法判断的部分，添加到中英文列表中
            chinese_parts.append(part)
            english_parts.append(part)
    
    chinese_text = " ".join(chinese_parts)
    english_text = " ".join(english_parts)
    
    return chinese_text, english_text

def match_ref_to_prediction(ref_path, poi_path, output_path, text_threshold=sim_name_threshold, distance_threshold=100):
    """
    将参考POI与预测POI进行匹配，并在参考POI中添加匹配信息
    支持中英文混合名称的匹配
    """
    # 读取参考POI数据集
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref_pois = ref_data.get("poi", [])
    
    # 读取预测POI数据集
    with open(poi_path, 'r', encoding='utf-8') as f:
        poi_data = json.load(f)
    
    # 为每个参考POI找到最佳匹配
    for ref_poi in ref_pois:
        best_match = None
        best_similarity = 0
        best_poi_id = None
        best_distance = None
        
        # 跳过没有名称或坐标的POI
        if 'name' not in ref_poi or not ref_poi['name']:
            continue
        if ref_poi["poi_id"] == "12771693772":
            print("#")
        # 处理参考POI名称，拆分中英文
        ref_raw_name = remove_parentheses(ref_poi['name'])
        ref_chinese, ref_english = split_chinese_english(ref_raw_name)
        
        # 标准化处理后的名称
        ref_name = normalize_text(ref_raw_name)
        ref_chinese = normalize_text(ref_chinese)
        ref_english = normalize_text(ref_english)
        
        ref_coord = [ref_poi.get('lat', 0), ref_poi.get('lon', 0)]
        
        # 遍历预测POI数据集
        for poi_id, poi in poi_data.items():
            # 检查预测POI是否有文本内容
            if 'name' not in poi:
                continue
                
            # 计算每个文本内容与参考POI名称的相似度，取最高值
            max_text_similarity = 0
            poi_text = normalize_text(poi['name'])
            
            # 分别计算与完整名称、中文部分和英文部分的相似度
            full_similarity = calculate_text_similarity(ref_name, poi_text)
            chinese_similarity = calculate_text_similarity(ref_chinese, poi_text)
            english_similarity = calculate_text_similarity(ref_english, poi_text)
            
            # 取三者最大值作为最终相似度
            text_similarity = max(full_similarity, chinese_similarity, english_similarity)
            max_text_similarity = max(max_text_similarity, text_similarity)
            
            # 如果文本相似度太低，跳过
            if max_text_similarity < text_threshold:
                continue
                
            # 计算坐标相似度和距离
            distance_similarity = 0
            distance = float('inf')
            
            if ('coordinate' in poi and len(poi['coordinate']) >= 2 and 
                ref_poi.get('lat') is not None and ref_poi.get('lon') is not None):
                poi_coord = poi['coordinate'][:2]  # 只取纬度和经度
                distance_similarity, distance = calculate_distance_similarity(ref_coord, poi_coord)
                
                # 如果距离太远，跳过
                if distance > distance_threshold:
                    continue
            
            # 计算最终相似度分数
            final_score = 0.4 * max_text_similarity + 0.6 * distance_similarity
            
            # 更新最佳匹配
            if final_score > best_similarity:
                best_similarity = final_score
                best_match = poi
                best_poi_id = poi_id
                best_distance = distance
        
        # 如果找到了最佳匹配且超过阈值
        if best_match and best_similarity > text_threshold:
            # 创建match_POI字段
            match_info = {
                "name": best_match.get("name", "") if best_match.get("name") else "",
                "poi_id": best_poi_id,
                "similarity": best_similarity,
                "coordinate": best_match.get("coordinate", [0, 0, 0])
            }
            
            # 添加距离信息（如果有）
            if best_distance != float('inf'):
                match_info["distance"] = best_distance
            
            # 添加match_POI字段
            ref_poi["match_POI"] = match_info
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ref_data, f, ensure_ascii=False, indent=4)
    
    return ref_data



def reverse_match_pois(poi_path,ref_path,output_path):
    """将参考POI匹配到预测POI并输出结果"""
    # 设置路径
    #args = config.parse_args()
    #poi_path = args.prediction_json
    #ref_path = "/home/moss/streetview_segment/poi_dataset/all/osm_poi_simplified.json"
    #output_path = args.reverse_match_json if hasattr(args, 'reverse_match_json') else os.path.join(os.path.dirname(ref_path), "matched_ref_poi.json")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 执行匹配
    groups_path = "/home/moss/streetview_segment/poi_dataset/all/poi_groups.json"

    
    match_ref_to_prediction(ref_path, poi_path, output_path)
    if groups_path:
        process_grouped_pois(output_path, groups_path)
    
    evaluate_matching_results(output_path)
    
    print(f"参考POI匹配完成！结果已保存到 {output_path}")

def evaluate_matching_results(output_path):
    """评估匹配结果，计算召回率和平均距离误差"""
    # 读取匹配结果
    with open(output_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref_pois = ref_data.get("poi", [])
    
    # 计算匹配成功的POI数量和总距离误差
    total_pois = len(ref_pois)
    matched_pois = 0
    total_distance_error = 0.0
    
    # 统计匹配成功的POI和距离误差
    for poi in ref_pois:
        if "match_POI" in poi:
            matched_pois += 1
            if "distance" in poi["match_POI"]:
                total_distance_error += poi["match_POI"]["distance"]
    
    # 计算召回率和平均距离误差
    recall_rate = matched_pois / total_pois if total_pois > 0 else 0
    avg_distance_error = total_distance_error / matched_pois if matched_pois > 0 else 0
    
    # 输出结果
    print(f"评估结果:")
    print(f"总POI数量: {total_pois}")
    print(f"匹配成功POI数量: {matched_pois}")
    print(f"召回率: {recall_rate:.4f} ({matched_pois}/{total_pois})")
    print(f"平均距离误差: {avg_distance_error:.4f} 米")
    
    plot_distance_boxplot(output_path)
    # 返回评估指标
    return {
        "total_pois": total_pois,
        "matched_pois": matched_pois,
        "recall_rate": recall_rate,
        "avg_distance_error": avg_distance_error
    }

def read_csv_to_dict(filename,type):
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        name_list = []
        poi_list = []
        for row in reader:
            name = row['\ufeff名称']
            name = convert(name, 'zh-hans')
            longitude = float(row['经度'])
            latitude = float(row['纬度'])
            type = row['类别']
            parts = type.split(';')
            type = parts[0]
            if type == 'gcj02':
                lat_wgs,lon_wgs = gcj02_to_wgs84(longitude,latitude)
                poi_dict = {
                    'name': name,
                    'lat_wgs': lat_wgs,
                    'lon_wgs': lon_wgs
                }
            else:
                poi_dict = {
                    'name': name,
                    'lat_wgs': latitude,
                    'lon_wgs': longitude,
                    'type': type
                }
            # 将每个POI的字典添加到列表中
            poi_list.append(poi_dict)
            name_list.append(name)
            #记得坐标系转换
            # location_dict[name] = [latitude,longitude]
    return poi_list,name_list

def read_json_to_dict(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def cal_sim_spatical(coordinate1_gps,coordinate_list_gps):
    coordinate1_ecef = gps2ecef(coordinate1_gps)
        # 将列表中的所有GPS坐标转换为ECEF坐标
    coordinate_list_ecef = np.array([gps2ecef(coord) for coord in coordinate_list_gps])
    # 计算单个坐标与列表中所有坐标之间的距离
    distances = np.linalg.norm(coordinate_list_ecef - coordinate1_ecef, axis=1)
    # 计算归一化距离
    normalized_distances = np.exp(-distances / spatial_threshold)
    return normalized_distances


def cal_similarity(doc_test, dictionary, tfidf, corpus):
    # 使用Levenshtein距离计算相似度
    corpus_text = [' '.join(dictionary.doc2bow(doc, return_words=True)) for doc in corpus]
    idx_similarity = []
    filtered_indices = []
    
    for idx, doc in enumerate(corpus_text):
        levenshtein_distance = Levenshtein.distance(doc_test, doc)
        normalized_levenshtein_distance = levenshtein_distance / max(len(doc_test), len(doc)) if max(len(doc_test), len(doc)) > 0 else 1
        similarity = 1 - normalized_levenshtein_distance
        
        if similarity > min_similarity:
            filtered_indices.append(idx)
            idx_similarity.append(similarity)
    
    return filtered_indices, idx_similarity

def remove_parentheses(text):
    # 检查字符串末尾是否有括号及其内容
    if text.endswith(')') and '(' in text:
        # 找到最后一个左括号的位置
        start_index = text.rfind('(')
        # 去掉括号及其内容
        text = text[:start_index]
    return text.strip()  # 去掉可能的多余空格

def cal_distance(ref_POI_list,eval_POI,similarity_list):
    coordinate0_gps = next(iter(eval_POI.values()))
    # coordinate0_ecef = gps2ecef(location_dict0[doc_test])
    coordinate0_ecef = gps2ecef(coordinate0_gps)
    distance_POI_pairs = []
    for i,POI_ref in enumerate(ref_POI_list):
        coordinate1_gps = [POI_ref['lat_wgs'], POI_ref['lon_wgs'], coordinate0_gps[2]]
        coordinate1_ecef = gps2ecef(coordinate1_gps)
        distance = np.linalg.norm(coordinate0_ecef - coordinate1_ecef)
        # 将距离和POI名存储为元组，添加到列表中
        distance_POI_pairs.append((distance, similarity_list[i], POI_ref))
    distance_POI_pairs.sort(key=lambda x: x[0])
    min_distance,ref_similarity, closest_POI = distance_POI_pairs[0]
    # 检查最小距离是否小于最大允许距离
    if min_distance < max_distance:
        # 记录最接近的匹配和距离
        return min_distance, ref_similarity, closest_POI
    else:
        return None,None, "No correspond POI"

def poi_compare(eval_POI_dict, ref_POI_list, name_list):
    ref_coordinate_list = [[poi['lat_wgs'],poi['lon_wgs'],0] for poi in ref_POI_list]
    matched_poi = {}
    
    for poi_id, value in eval_POI_dict.items():
        eval_name = value['name']
        filtered_indices = []
        sim_name = []
        
        # 使用normalized Levenshtein距离计算文本相似度
        for idx, ref_name in enumerate(name_list):
            ref_name = remove_parentheses(ref_name)
            levenshtein_distance = Levenshtein.distance(eval_name, ref_name)
            normalized_levenshtein_distance = levenshtein_distance / max(len(eval_name), len(ref_name)) if max(len(eval_name), len(ref_name)) > 0 else 1
            similarity = 1 - normalized_levenshtein_distance
            
            if similarity > sim_name_threshold:
                filtered_indices.append(idx)
                sim_name.append(similarity)
        
        # 剩余处理保持不变
        if filtered_indices:
            filtered_ref_coordinates = [ref_coordinate_list[index] for index in filtered_indices]
            coordinate_wgs = [value['coordinate'][0],value['coordinate'][1],0]
            sim_distance = cal_sim_spatical(coordinate_wgs, filtered_ref_coordinates)
            sim_sum = 0.6 * np.array(sim_name) + 0.4 * sim_distance
            max_sim = sim_sum[int(np.argmax(sim_sum))]
            match_idx = filtered_indices[int(np.argmax(sim_sum))]
            
            # 以下代码保持不变
            coordinate1_ecef = gps2ecef(value['coordinate'])
            coordinate2_ecef = gps2ecef(ref_coordinate_list[match_idx])
            distance = np.linalg.norm(coordinate1_ecef - coordinate2_ecef)
            if distance<spatial_threshold:
                matched_poi[poi_id] = {
                    'name': eval_name,
                    'coordinate': value['coordinate'],
                    'type': ref_POI_list[match_idx]['type'],
                    'match_POI': name_list[match_idx],
                    'match_idx': match_idx,
                    'match_coordinate': [ref_POI_list[match_idx]['lat_wgs'],ref_POI_list[match_idx]['lon_wgs']],
                    'distance':distance,
                    'similarity': max_sim
                }
            else:
                matched_poi[poi_id] = {
                    'name': eval_name,
                    'coordinate': value['coordinate'],
                    'type': '',
                    'match_POI': '',
                    'match_idx': '',
                    'match_coordinate': '',
                    'distance': '',
                    'similarity': ''
                }
        else:
            matched_poi[poi_id] = {
                'name': eval_name,
                'coordinate': value['coordinate'],
                'type': '',
                'match_POI': '',
                'match_idx': '',
                'match_coordinate': '',
                'distance': '',
                'similarity': ''
            }
    return matched_poi

def calculate_text_similarity(text1, text2):
    """计算文本相似度"""
    levenshtein_distance = Levenshtein.distance(text1, text2)
    normalized_levenshtein_distance = levenshtein_distance / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 1
    return 1 - normalized_levenshtein_distance

def calculate_distance_similarity(coord1, coord2, max_distance=spatial_threshold):
    """计算距离相似度"""
    # 使用haversine公式计算两个坐标点之间的距离（单位：米）
    distance = haversine((coord1[0], coord1[1]), (coord2[0], coord2[1]), unit=Unit.METERS)
    # 距离越小，相似度越高，线性归一化到[0,1]区间
    similarity = max(0, 1 - distance / max_distance)
    return similarity, distance

def normalize_text(text):# -> LiteralString | Any:
    """标准化文本，移除空格和转为小写"""
    if isinstance(text, list):
        text = " ".join(text)
    return text

def find_best_match(poi, ref_pois, text_threshold=0.3, distance_threshold=100):
    """为给定POI找到最佳匹配的参考POI
    先按名称相似度筛选，再按距离筛选，最后按综合相似度排序
    """
    candidates = []
    
    # 获取待匹配POI的文本和坐标
    poi_text = normalize_text(poi["name"])
    poi_coord = poi["coordinate"][:2]  # 只取纬度和经度
    
    # 第一步：计算名称相似度，筛选出超过阈值的候选
    for ref_poi in ref_pois:
        ref_text = normalize_text(ref_poi["name"])
        
        # 计算文本相似度
        text_similarity = calculate_text_similarity(poi_text, ref_text)
        
        # 只有名称相似度超过阈值的才进入下一步
        if text_similarity > text_threshold:
            candidates.append({
                "poi": ref_poi,
                "text_similarity": text_similarity,
                "distance": None,
                "distance_similarity": 0,
                "final_score": text_similarity  # 初始分数就是文本相似度
            })
    
    # 如果没有候选，直接返回None
    if not candidates:
        return None, 0, None
    
    # 第二步：计算距离和距离相似度
    valid_candidates = []
    for candidate in candidates:
        ref_poi = candidate["poi"]
        
        # 只对有坐标的参考POI计算距离
        if ref_poi.get("match_type") == "GD" and "lat" in ref_poi and "lon" in ref_poi:
            ref_coord = [ref_poi["lat"], ref_poi["lon"]]
            distance_similarity, distance = calculate_distance_similarity(poi_coord, ref_coord)
            candidate["distance"] = distance
            candidate["distance_similarity"] = distance_similarity
            
            # 更新最终分数
            candidate["final_score"] = 0.6 * candidate["text_similarity"] + 0.4 * distance_similarity
            
            # 只有距离小于阈值的才是有效候选
            if distance < distance_threshold:
                valid_candidates.append(candidate)
        else:
            # 对于没有坐标的POI，保留其文本相似度作为最终分数
            valid_candidates.append(candidate)
    
    # 如果没有有效候选（即所有候选都超过距离阈值），返回None
    if not valid_candidates:
        return None, 0, None
    
    # 第三步：按最终分数排序，选择最高分
    best_candidate = max(valid_candidates, key=lambda x: x["final_score"])
    best_match = best_candidate["poi"]
    best_score = best_candidate["final_score"]
    best_distance = best_candidate["distance"]
    
    return best_match, best_score, best_distance

def match_pois(poi_path, ref_path, output_path, mismatch_path):
    """匹配两个POI数据集并输出结果"""
    # 读取待匹配POI数据集
    with open(poi_path, 'r', encoding='utf-8') as f:
        poi_data = json.load(f)
    
    # 读取参考POI数据集
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
        ref_pois = ref_data.get("poi", [])
    
    # 处理参考POI的名称，去掉括号内容
    for poi in ref_pois:
        if 'name' in poi:
            poi['name'] = remove_parentheses(poi['name'])

    # 打开不匹配记录文件
    mismatch_file = open(mismatch_path, 'w', encoding='utf-8')
    
    # 遍历每个待匹配POI
    for poi_id, poi in poi_data.items():
        # 找到最佳匹配的参考POI
        best_match, similarity, distance = find_best_match(poi, ref_pois)
        
        if best_match:
            # 创建match_POI字段
            if best_match.get("match_type") == "GD" and "lat" in best_match and "lon" in best_match:
                match_info = {
                    "type": best_match.get("type", "未知"),
                    "name": best_match.get("name", ""),
                    "poi_id": best_match.get("poi_id", -1),
                    "coordinate": [best_match.get("lat", 0), best_match.get("lon", 0)],
                    "distance": distance,
                    "similarity": similarity
                }
            else:
                match_info = {
                    "type": best_match.get("type", "manual"),
                    "name": best_match.get("name", ""),
                    "poi_id": best_match.get("poi_id", -1),
                    "similarity": similarity
                }
            
            # 添加match_POI字段
            poi["match_POI"] = match_info
            
            # 检查匹配的POI是否与label字段中的POI相同
            if "label" in poi and best_match.get("poi_id") != poi["label"].get("poi_id"):
                mismatch_info = f"POI ID: {poi_id}, 匹配到: {best_match.get('name')}(ID:{best_match.get('poi_id')}), "
                mismatch_info += f"但label为: {poi['label'].get('name')}(ID:{poi['label'].get('poi_id')}), "
                mismatch_info += f"相似度: {similarity:.4f}"
                mismatch_file.write(mismatch_info + '\n')
    
    # 保存匹配结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(poi_data, f, ensure_ascii=False, indent=4)
    
    # 关闭不匹配记录文件
    mismatch_file.close()

if __name__ == "__main__":
    # 设置路径
    args = config.parse_args()
    poi_path = args.prediction_json
    ref_path = "/home/moss/streetview_segment/poi_dataset/all/osm_poi_simplified.json"
    output_path = args.match_json
    mismatch_path = args.mismatch_path
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 执行匹配
    #match_pois(poi_path, ref_path, output_path, mismatch_path)
    #print(f"匹配完成！结果已保存到 {output_path}")
    #print(f"不匹配label的记录已保存到 {mismatch_path}")
    reverse_match_pois(poi_path,ref_path,output_path)

