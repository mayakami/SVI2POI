# -*- coding: utf-8 -*-
from pathlib import Path
import sqlite3
import sys
import shutil
import json
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2 as cv
import os
from collections import OrderedDict
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from unionfind import UnionFind
from database import array_to_blob,blob_to_array
import matplotlib.pyplot as plt
from read_file import read_file
import pickle
import csv
from gensim import corpora
from gensim import models
from gensim import similarities
import jieba
import csv
import os
from geocoding import gcj02_to_wgs84,wgs84_to_gcj02
from ecef2gps import gps2ecef,ecef2gps
import numpy as np
from zhconv import convert
import config
from unionfind import UnionFind
import zhconv
import Levenshtein
spatial_threshold = 100
from math import exp

from create_sign import sign_ids_to_pair_id,pair_id_to_sign_ids,SIGNDatabase
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ecef2gps'))
if module_path not in sys.path:
    sys.path.append(module_path)
from ecef2gps import ecef2gps, gps2ecef
import config
from sklearn.cluster import DBSCAN

torch.set_grad_enabled(False)
IS_PYTHON3 = sys.version_info[0] >= 3
#距离阈值
max_distance = 100
#相似度阈值
min_similarity =0.6
query_sign = '''
SELECT sign_id, x_min, y_min, x_max, y_max, text_content, image_name
FROM signs
WHERE sign_id IN (?, ?)
ORDER BY sign_id
'''

query_kp = '''
        SELECT image_name,data
        FROM keypoints
        WHERE image_name IN (?, ?)
        ORDER BY image_name
        '''

CREATE_POI_TABLE = '''
CREATE TABLE IF NOT EXISTS poi (
    poi_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, 
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    alt REAL NOT NULL,
    text_content TEXT,
    text_confidence REAL,
    image_name TEXT,
    points_num INTEGER NOT NULL,
    sign_id INTEGER NOT NULL
)
'''

CREATE_ALL = "; ".join(
    [
        CREATE_POI_TABLE
    ]
)


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :] #[a,b,c,d]

def is_inlier(plane, xyz, threshold):
    plane_normal_magnitude = np.linalg.norm(plane[:3])
    distance = np.abs(plane.dot(augment([xyz]).T)) / plane_normal_magnitude
    return distance < threshold

def run_ransac(data, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    best_inliers = []
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size)) #随机取出三个点
        m = estimate(s)
        ic = 0
        inliers = []
        for j in range(len(data)):
            if is_inlier(m, data[j],0.3):
                ic += 1
                inliers.append(data[j])
        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_inliers = inliers
            if ic > goal_inliers and stop_at_goal:
                break
    # print('took iterations:', i+1, 'best model:', best_model, 'inliers:', best_ic)
    return best_model, best_inliers

def read_points3d(db):
    points3d = dict(
        (point3d_id, pickle.loads(data))
        for point3d_id, data in db.execute("SELECT point3d_id, data FROM points3d")
    )
    # 获取所有点的坐标数据
    points_data = list(points3d.values())
    # 获取点的索引列表
    point_indices = list(points3d.keys())
    # 创建一个大小与点的数量相同的 NumPy 数组，初始化为 -1
    max_index = max(point_indices)
    points_list = [-1 for index in range(int(max_index) + 1)]
    # 将每个点的数据放入对应的索引位置
    for index, data in zip(point_indices, points_data):
        points_list[index] = data
    return points_list

def cal_plane(sign,points3d):
    points3d_plane = []
    for idx in sign['correspond_points3d_idx']:
        # 如果 idx 存在于 points3d 中，则将对应的点数据添加到 points3d_plane 列表中
        points3d_plane.append(points3d[idx])
    points3d_plane = np.array(points3d_plane) #n*3
    if len(points3d_plane)>3:
        plane,inliers = run_ransac(points3d_plane,3,int(0.3*len(points3d_plane)),100) #平面方程,内点坐标
        if inliers:
            center = np.mean(inliers,axis=0)
        else:
            center = np.mean(points3d_plane,axis=0)
    else:
        plane = None
        center = np.mean(points3d_plane,axis=0)
    return plane,center

class POIdatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=POIdatabase) #生成自定义类型的实例
    def __init__(self, *args, **kwargs):
        super(POIdatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_poi_table = lambda: self.executescript(  # executescript：执行CREATE_SIGNES_TABLE命令
            CREATE_POI_TABLE
        )
    def add_poi(self,poi,poi_id = None):
        lat,lon,alt = poi['coordinate']
        # text_content = sign_value['text']
        image_name = poi['image_name']
        points_num = poi['points_num']
        text_content = poi['text_content']
        text_confidence = poi['text_confidence']
        sign_id = poi['sign_id']
        curor = self.execute(
            "INSERT INTO poi VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (poi_id,lat,lon,alt,text_content,text_confidence,image_name,points_num,sign_id))
        return curor.lastrowid

def cal_coordinate(top_k_signs,points3d):
    for sign_id, value in top_k_signs.items():
        plane, center = cal_plane(value, points3d)  # 平面法向量，中心坐标 in ecef坐标系
        if np.any(np.isnan(center)):
            continue
        else:
            # coordinate = ecef2gps(center) #转换到经纬度坐标系
            top_k_signs[sign_id]['coordinate'] = center
    coordinates = [item.get('coordinate') for item in top_k_signs.values() if item.get('coordinate') is not None]
    if coordinates != []:
        coordinates = np.array(coordinates)
        mean_coordinate = np.mean(coordinates, axis=0)
        # coordinate_gps = ecef2gps(mean_coordinate)
        std_dev = np.std(coordinates, axis=0)
        threshold = 2 * std_dev
        filtered_coordinates = coordinates[np.all(np.abs(coordinates - mean_coordinate) <= threshold, axis=1)]
        final_mean_coordinate = np.mean(filtered_coordinates, axis=0)
        final_coordinate_gps = ecef2gps(final_mean_coordinate)
        return final_coordinate_gps
    else:
        return None

def normalized_Levenshtein_distance(text1, text2):
    min_normalized_distance = float('inf')
    if type(text1) == list and type(text2) == list:
        for str1 in text1:
            for str2 in text2:
                distance = Levenshtein.distance(str1, str2)
                # 计算归一化Levenshtein距离
                normalized_distance = distance / max(len(str1), len(str2))
                # 更新最小归一化Levenshtein距离
                if normalized_distance < min_normalized_distance:
                    min_normalized_distance = normalized_distance
    elif type(text1) == list and type(text2) == str:
        for str1 in text1:
            distance = Levenshtein.distance(str1, text2)
            normalized_distance = distance / max(len(str1), len(text2))
            if normalized_distance < min_normalized_distance:
                min_normalized_distance = normalized_distance

    elif type(text2) == list and type(text1) == str:
        for str2 in text2:
            distance = Levenshtein.distance(str2, text1)
            normalized_distance = distance / max(len(str2), len(text1))
            if normalized_distance < min_normalized_distance:
                min_normalized_distance = normalized_distance
    elif type(text1) == str and type(text2) == str:
        distance = Levenshtein.distance(text1, text2)
        min_normalized_distance = distance / max(len(text1), len(text2))
    return min_normalized_distance

#输入是[[str1,str2],……]
def combine_text(texts):
    # 合并所有文本列表，并去除重复的单词
    combined_words = set()
    for sublist in texts:
        for word in sublist:
            word = zhconv.convert(word, 'zh-hans')
            combined_words.add(word)
    combined_words_copy = combined_words.copy()
    for word in combined_words_copy:
        for other_word in combined_words_copy:
            if word != other_word and word in other_word:
                combined_words.discard(word)
    combined_words = list(combined_words)
    return combined_words

def create_poi(args,group_id):
    """
    根据sign.db数据库创建POI,将关联的标识牌分组，计算每组的位置坐标和合并文本内容，
    
    参数:
        args: 配置参数对象，包含以下字段:
            - group_path: 组数据根目录
            - sign_path: 标识牌数据库的相对路径
            - k: 聚类数 (可选，默认为4)，用于选择每组中的顶部标识牌数量
            - match_threshold: 匹配分数阈值 (可选，默认为0.2)，只有超过此阈值的匹配才被认为有效
        
        group_id: 字符串，表示当前处理的组ID
        
    返回:
        POI_list: 列表，包含生成的POI数据，每个POI是一个字典，包含以下字段:
            - text_content: 列表，POI的文本内容
            - coordinate: 列表 [纬度, 经度, 高度]，POI的地理坐标
            - sign_id: 列表，与POI关联的所有标识牌ID
            - top_k_sign_id: 列表，与POI关联的前k个标识牌ID（按3D点数量排序）
    """
    #解析参数
    group_dir = os.path.join(args.group_path, group_id)
    sign_db = os.path.join(group_dir, args.sign_path)
    # 检查数据库文件是否存在
    if not os.path.exists(sign_db):
        return []
    k = args.k if hasattr(args, 'k') else 4
    match_threshold = args.match_threshold if hasattr(args, 'match_threshold') else 0.2
    
    # Open the database.
    db = SIGNDatabase.connect(sign_db)
    cursor = db.cursor()
    # 检查表结构是否包含match_score列
    has_match_score = False
    cursor.execute("PRAGMA table_info(matches)")
    columns = cursor.fetchall()
    for column in columns:
        if column[1] == 'match_score':
            has_match_score = True
            break
    
    # 根据表结构选择合适的查询语句
    if has_match_score:
        try:
            cursor.execute("""
                SELECT sign_id1, sign_id2, match_score FROM matches 
                WHERE match_score >= ?
            """, (match_threshold,))
            match_ids = [(sign_id1, sign_id2) for sign_id1, sign_id2, match_score in cursor.fetchall() 
                     if match_score >= match_threshold]
        except sqlite3.OperationalError:
            # 如果查询失败，回退到不使用match_score的查询
            cursor.execute("SELECT sign_id1, sign_id2 FROM matches")
            match_ids = [(sign_id1, sign_id2) for sign_id1, sign_id2 in cursor.fetchall()]
    else:
        # 直接查询所有匹配关系
        cursor.execute("SELECT sign_id1, sign_id2 FROM matches")
        match_ids = [(sign_id1, sign_id2) for sign_id1, sign_id2 in cursor.fetchall()]
    
    points3d = read_points3d(db)
    sign_ids = [row[0] for row in db.execute("SELECT sign_id FROM signs")]

    #构建并查集, 关联的招牌ID分组。匹配的招牌会被合并到同一组。
    uf = UnionFind()
    for sign_id in sign_ids:
        uf.add(sign_id)
    for sign1, sign2 in match_ids:
        uf.union(sign1, sign2)
    #获得并查集
    groups = uf.get_groups()
    POIs = []
    #对每个分组，查询数据库获取招牌的详细信息,放在sign_group中{sign_id:{xxx}}
    #按关联的3D点数量排序，取前k个招牌

    for key,group in groups.items():
        sign_group = {}
        # 读取signs
        query = "SELECT sign_id, x_min, y_min, x_max, y_max, text_content, text_confidence, image_name, correspond_points3d_idx FROM signs WHERE sign_id IN ({})".format(
            ','.join('?' * len(group)))
        rows = db.execute(query, group).fetchall()
        for row in rows:
            sign_id, x_min, y_min, x_max, y_max, text_content, text_confidence, image_name, data = row
            bbox = np.array([x_min, y_min, x_max, y_max])
            correspond_points3d_idx = pickle.loads(data)
            sign_group[sign_id] = {
                'bbox': bbox,
                'text_content': json.loads(text_content),
                'text_confidence': text_confidence,
                'image_name': image_name,
                'correspond_points3d_idx': correspond_points3d_idx
            }

        #多招牌处理，按关联的3D点数量排序，取前5个招牌 top_k_signs={sign_id:{xxx}}
        #合并文本内容， 计算GPS坐标
        if len(sign_group) >1: #至少包含两个招牌图像
            #先计算位置,根据correspond_points3d_idx数量排序,选择top k个sign进行位置计算
            sorted_sign_group = OrderedDict(
                sorted(sign_group.items(), key=lambda x: len(x[1]['correspond_points3d_idx']), reverse=True))
            top_k_signs = list(sorted_sign_group.items())[:k]
            top_k_signs = {sign[0]:sign[1] for sign in top_k_signs}
            #提取text组成一个字符串
            texts = [item['text_content'] for item in top_k_signs.values()]
            combined_text = combine_text(texts)
            #计算位置
            final_coordinate_gps = cal_coordinate(top_k_signs,points3d)
            if final_coordinate_gps and combined_text!=[]:
                POIs.append({
                    'text_content': combined_text,
                    'coordinate': final_coordinate_gps,
                    'sign_id': list(sign_group.keys()),
                    'top_k_sign_id': list(top_k_signs.keys())
                })
        else:
            #只以一个招牌图像计算plane和center
            # for key, value in sign_group.items():
            #     print(value['text_content'])
            plane, center = cal_plane(sign_group[sign_id], points3d)
            if np.any(np.isnan(center)):
                continue
            else:
                coordinate = ecef2gps(center)  # (lat,lon,alt)
                for key, value in sign_group.items():
                    if value['text_content'] != []:
                        POIs.append({
                            'text_content': [zhconv.convert(text, 'zh-hans') for text in value['text_content']],
                            'coordinate': coordinate,
                            'sign_id': [key],
                            'top_k_sign_id': [key]
                        })
    #save poi
    for POI in POIs:
        db.add_poi(
            text_content = POI['text_content'],
            coordinate = POI['coordinate'],
            sign_id = POI['sign_id'],
            top_k_sign_id = POI['top_k_sign_id']
        )
    db.commit()
    #read and check
    pois = {
        poi_id: {
            'text_content': json.loads(text_content),
            'coordinate': json.loads(coordinate),
            'sign_id': json.loads(sign_id),
            'top_k_sign_id': json.loads(top_k_sign_id)
        }
        for poi_id,text_content,coordinate,sign_id,top_k_sign_id in
        db.execute("SELECT poi_id, text_content, coordinate, sign_id, top_k_sign_id FROM pois")
    }
    db.close()
    return POIs

def similarity(doc_test,doc_test_idx, dictionary, tfidf, corpus):
    doc_test_list = [word for word in jieba.cut_for_search(doc_test)]
    doc_test_vec = dictionary.doc2bow(doc_test_list)
    # 创建相似度矩阵
    sparse_matrix = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sim = sparse_matrix.get_similarities(tfidf[doc_test_vec])

    # 排除测试文档本身的比较
    filtered_indices = [idx for idx, similarity in enumerate(sim) if similarity > min_similarity and idx != doc_test_idx]
    return filtered_indices

def cal_sim_spatical(coordinate1_gps,coordinate2_gps):
    coordinate1_ecef = gps2ecef(coordinate1_gps)
    coordinate2_ecef = gps2ecef(coordinate2_gps)
    distance = np.linalg.norm(coordinate1_ecef - coordinate2_ecef)
    if distance>spatial_threshold:
        sim_spatical = 0
    else:
        sim_spatical = exp(-distance/spatial_threshold)
    return sim_spatical

def read_poi(sign_db):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default=sign_db)
    args = parser.parse_args()
    if not os.path.exists(args.database_path):
        print("ERROR: SIGN Database does not exist.")
        return
    db = SIGNDatabase.connect(args.database_path)
    POIs = [
        {
            'text_content': json.loads(text_content),
            'coordinate': json.loads(coordinate),
            'sign_id': json.loads(sign_id),
            'top_k_sign_id': json.loads(top_k_sign_id)  # 这个地方一定会解析为list,不管有没有元素
        }
        for text_content, coordinate, sign_id, top_k_sign_id in
        db.execute("SELECT text_content, coordinate, sign_id, top_k_sign_id FROM pois")
    ]
    return POIs

def filt_poi(sign_db):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default=sign_db)
    args = parser.parse_args()
    if not os.path.exists(args.database_path):
        print("ERROR: SIGN Database does not exist.")
        return
    # Open the database.
    db = SIGNDatabase.connect(args.database_path)

    #read and check
    POIs = [
        {
            'text_content': json.loads(text_content),
            'coordinate': json.loads(coordinate),
            'sign_id': json.loads(sign_id),
            'top_k_sign_id': json.loads(top_k_sign_id) #这个地方一定会解析为list,不管有没有元素
        }
        for text_content,coordinate,sign_id,top_k_sign_id in
        db.execute("SELECT text_content, coordinate, sign_id, top_k_sign_id FROM pois")
    ]

    uf = UnionFind()
    name_list = []
    filted_POIs = []
    for i,POI in enumerate(POIs):
        uf.add(i)
        name_list.append(POI['text_content'])

    for i,text1 in enumerate(name_list):
        text1 = ' '.join(text1)
        max_sim = 0
        for j in range(i + 1, len(name_list)):  # 从i+1开始遍历
            text2 = name_list[j]
            text2 = ' '.join(text2)
            sim_spatical = cal_sim_spatical(POIs[i]['coordinate'],POIs[j]['coordinate'])
            levenshtein_distance = Levenshtein.distance(text1, text2)
            normalized_levenshtein_distance = levenshtein_distance / max(len(text1), len(text2))
            # normalized_levenshtein_distance = normalized_Levenshtein_distance(text1, text2)
            sim_name = 1 - normalized_levenshtein_distance
            sim_sum = 0.5*sim_name + 0.5*sim_spatical
            if sim_sum>max_sim:
                max_sim = sim_sum
                match_index = j
        if max_sim>0.65:
            text2 = name_list[match_index]
            print("{}\t{}\t{}".format(text1, text2, max_sim))
            uf.union(i, match_index)

    groups = uf.get_groups()
    for key, group in groups.items():
        POI_group = [POIs[i] for i in group]
        if len(group) > 1:  #至少包含两个相同的POI
            #sign_id
            combined_sign_id = []
            combined_topk_sign_id = []
            for item in POI_group:
                if isinstance(item.get('sign_id'), list):
                    combined_sign_id.extend(item['sign_id'])
                elif item.get('sign_id'):
                    combined_sign_id.append(item['sign_id'])

                if isinstance(item.get('top_k_sign_id'), list):
                    combined_topk_sign_id.extend(item['top_k_sign_id'])
                elif item.get('top_k_sign_id'):
                    combined_topk_sign_id.append(item['top_k_sign_id'])
            #位置
            coordinates = np.array([gps2ecef(POI_group[i]['coordinate']) for i in range(len(group))])
            mean_coordinate = np.mean(coordinates, axis=0) 
            coordinate_gps = ecef2gps(mean_coordinate)
            std_dev = np.std(coordinates, axis=0)
            threshold = std_dev
            filtered_coordinates = coordinates[np.all(np.abs(coordinates - mean_coordinate) <= threshold, axis=1)]
            final_mean_coordinate = np.mean(filtered_coordinates, axis=0)
            final_coordinate_gps = ecef2gps(final_mean_coordinate)
            text_list = [POI_group[i]['text_content'] for i in range(len(POI_group))]
            combined_words = combine_text(text_list) 
            # combined_words = set()
            filted_POIs.append({
                'text_content': combined_words,
                'coordinate': final_coordinate_gps,
                'sign_id': combined_sign_id,
                'top_k_sign_id': combined_topk_sign_id
            })
        else:
            filted_POIs.extend(POI_group)
    return filted_POIs


if __name__ == '__main__':
    args = config.parse_args()
    group_path = args.group_path
    groups = os.listdir(group_path)
    for group_id in groups:
        print(group_id)
        sign_path = r"{}\{}\{}".format(group_path, group_id, args.sign_path)
        poi_path = r"{}\{}\{}".format(group_path, group_id, args.poi_path)
        csv_file = r"{}\{}\{}".format(group_path, group_id, args.poi_csv)
        POIs = create_poi(sign_path) #读取db_path,并从中计算POI点
        filted_POIs = filt_poi(POIs)
        save_poi(filted_POIs,poi_path,csv_file) #从db文件中读取poi，然后存入csv文件中

















