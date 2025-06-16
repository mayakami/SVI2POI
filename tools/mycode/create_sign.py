# -*- coding: utf-8 -*-
from pathlib import Path
import sqlite3
import pickle
import sys
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
from database import array_to_blob,blob_to_array
import matplotlib.pyplot as plt
from read_file import read_file
import config_signboard_entity_area1 as config
import json
torch.set_grad_enabled(False)
MAX_IMAGE_ID = 2 ** 31 - 1
IS_PYTHON3 = sys.version_info[0] >= 3

# 创建表
CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    correspond_struct_idx BLOB
)
"""

CREATE_SIGNES_TABLE = '''
CREATE TABLE IF NOT EXISTS signs (
    sign_id INTEGER NOT NULL, 
    x_min INTEGER NOT NULL,
    y_min INTEGER NOT NULL,
    x_max INTEGER NOT NULL,
    y_max INTEGER NOT NULL,
    text_content TEXT,
    text_confidence REAL,
    image_name TEXT,
    correspond_points3d_idx BLOB
)
'''
#存储匹配点集的idx
CREATE_MATCHES_TABLE = '''
CREATE TABLE IF NOT EXISTS matches (
    sign_id1 INTEGER NOT NULL,
    sign_id2 INTEGER NOT NULL,
    match_score REAL DEFAULT 0.0,
    data BLOB
)
'''
#存储图像中所有特征点坐标
CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_name TEXT PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
)
"""

CREATE_POINTS3D_TABLE = '''
CREATE TABLE IF NOT EXISTS points3d (
    point3d_id INTEGER PRIMARY KEY NOT NULL, 
    data BLOB
)
'''

CREATE_POIS_TABLE = '''
CREATE TABLE IF NOT EXISTS pois (
    poi_id INTEGER PRIMARY KEY NOT NULL, 
    text_content TEXT,
    coordinate TEXT,
    sign_id TXET,
    top_k_sign_id TEXT
)
'''

CREATE_ALL = "; ".join(
    [
        CREATE_SIGNES_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_POINTS3D_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POIS_TABLE
    ]
)

#如何存储：
def sign_ids_to_pair_id(sign_id1, sign_id2):
    if sign_id1 > sign_id2:
        sign_id1, sign_id2 = sign_id2, sign_id1
    return sign_id1 * MAX_IMAGE_ID + sign_id2

def pair_id_to_sign_ids(pair_id):
    sign_id2 = pair_id % MAX_IMAGE_ID
    sign_id1 = (pair_id - sign_id2) / MAX_IMAGE_ID
    return int(sign_id1), int(sign_id2)

def is_point_in_bbox(point, bbox):
    y,x = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def feature_to_bbox(features,bboxes):
    feature_to_bbox_map = [-1] * len(features)
    for i, bbox in enumerate(bboxes):
        for feature_idx, feature_point in enumerate(features):
            if is_point_in_bbox(feature_point, bbox):
                feature_to_bbox_map[feature_idx] = i
    return feature_to_bbox_map

def find_corresponding_sign(feature_matches, features_img1, features_img2, data1, data2):
    corresponding_idx = []
    corresponding_ids = []
    if features_img1 is None or features_img2 is None:
        return corresponding_ids,corresponding_idx
    else:
        match_counts = {sign_id: 0 for sign_id in data2}
        i = 0
        for sign1 in data1:
            bbox1 = data1[sign1]["board_contour"]
            match_idxs = {sign_id: [] for sign_id in data2}
            count0 = 0
            for feature_idx1, feature_idx2 in feature_matches:
                feature_point_img1 = features_img1[feature_idx1]
                feature_point_img2 = features_img2[feature_idx2]
                if is_point_in_bbox(feature_point_img1, bbox1):
                    count0 += 1
                for sign2 in data2:
                    bbox2= data2[sign2]["board_contour"]
                    if is_point_in_bbox(feature_point_img2,bbox2):
                        match_counts[sign2] += 1
                        match_idxs[sign2].append([feature_idx1, feature_idx2])

            max_matches = max(match_counts.values())
            if max_matches > max(0.5 * count0, 4):
                corresponding_id = max(match_counts, key=match_counts.get)
                corresponding_ids.append([sign1,corresponding_id])
                corresponding_idx.append(np.array(match_idxs[corresponding_id]))
        return corresponding_ids,corresponding_idx

            #     corresponding_bboxes.append((bbox1, bboxes_img2[corresponding_bbox_idx]))
            #     corresponding_idx.append(np.array(match_idxs[corresponding_bbox_idx]))
            # return corresponding_bboxes, corresponding_id


            # if match_counts:
            #     max_matches = max(match_counts.values())
            #     if max_matches > max(0.5 * count0, 4):  # 如果存在有效匹配
            #         corresponding_bbox_idx = max(match_counts, key=match_counts.get)
            #         corresponding_bboxes.append((bbox1, bboxes_img2[corresponding_bbox_idx]))
            #         corresponding_idx.append(np.array(match_idxs[corresponding_bbox_idx]))


def find_corresponding_bbox(feature_matches, features_img1, features_img2, bboxes_img1, bboxes_img2):
    corresponding_bboxes = []
    corresponding_idx = []
    if features_img1 is None or features_img2 is None or bboxes_img1 is None or bboxes_img2 is None:
        return corresponding_bboxes, corresponding_idx
    else:
        for bbox1 in bboxes_img1:
            match_counts = {i: 0 for i in range(len(bboxes_img2))} #为图2的每个框初始化计数
            match_idxs ={i: [] for i in range(len(bboxes_img2))}
            count0 = 0
            for feature_idx1, feature_idx2 in feature_matches:
                feature_point_img1 = features_img1[feature_idx1]
                feature_point_img2 = features_img2[feature_idx2]
                if is_point_in_bbox(feature_point_img1, bbox1):
                    count0 += 1
                    # 计算图2中匹配特征点落在哪个检测框内
                    for i, bbox2 in enumerate(bboxes_img2):
                        if is_point_in_bbox(feature_point_img2, bbox2):
                            match_counts[i] += 1
                            match_idxs[i].append([feature_idx1, feature_idx2])
            # 确定可能的对应检测框
            if match_counts:
                max_matches = max(match_counts.values())
                if max_matches > max(0.5*count0,4):  # 如果存在有效匹配
                    corresponding_bbox_idx = max(match_counts, key=match_counts.get)
                    corresponding_bboxes.append((bbox1, bboxes_img2[corresponding_bbox_idx]))
                    corresponding_idx.append(np.array(match_idxs[corresponding_bbox_idx]))
        return corresponding_bboxes,corresponding_idx

class SIGNDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=SIGNDatabase) #生成自定义类型的实例
    def __init__(self, *args, **kwargs):
        super(SIGNDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_signes_table = lambda: self.executescript(  #executescript：执行CREATE_SIGNES_TABLE命令
            CREATE_SIGNES_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_points3d_table = lambda: self.executescript(
            CREATE_POINTS3D_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_poi_table = lambda: self.executescript(
            CREATE_POIS_TABLE
        )

    def add_sign(self,sign_id,x_min,y_min,x_max,y_max,correspond_points3d_idx = None,text_content = None,text_confidence = None,image_name = None):
        '''
        子表 signs 
        sign_id TEXT NOT NULL, 
        x_min INTEGER NOT NULL,
        y_min INTEGER NOT NULL,
        x_max INTEGER NOT NULL,
        y_max INTEGER NOT NULL,
        text_content TEXT,
        text_confidence REAL,
        image_name TEXT,
        correspond_points3d_idx BLOB
        '''
        #检查是否已经存入过：
        cursor = self.execute(
            "SELECT sign_id FROM signs WHERE sign_id=?",
            (sign_id,))
        existing = cursor.fetchone()
        if existing:
            return existing[0]

        data_blob = pickle.dumps(correspond_points3d_idx)
        cursor = self.execute(
            "INSERT INTO signs VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)",
            (sign_id,x_min,y_min,x_max,y_max,text_content,text_confidence,image_name,data_blob))
        return cursor.lastrowid

    def add_matches(self,sign_id1,sign_id2,matches, match_score=0.0):
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (sign_id1,sign_id2,match_score,array_to_blob(matches)))

    def add_keypoints(self, image_name, keypoints):
        '''
        子表 keypoints
        image_name TEXT PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
        '''
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]
        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_name,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_points3d(self,points3d):
        '''
        子表 points3d
        point3d_id INTEGER PRIMARY KEY NOT NULL, 
        data BLOB
        '''
        # for point
        for key, value in points3d.items():
            data_blob = pickle.dumps(value)
            self.execute(
                "INSERT INTO points3d VALUES (?, ?)",
                (key,data_blob),
            )

    def add_image(
        self,image_id,name,correspond_struct_idx,prior_q=np.full(4, np.NaN),prior_t=np.full(3, np.NaN),
    ):
        '''
        子表 images
        image_id INTEGER PRIMARY KEY NOT NULL,
        name TEXT NOT NULL UNIQUE,
        prior_qw REAL,
        prior_qx REAL,
        prior_qy REAL,
        prior_qz REAL,
        prior_tx REAL,
        prior_ty REAL,
        prior_tz REAL,
        correspond_struct_idx BLOB
        '''
        data_blob = pickle.dumps(correspond_struct_idx)
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,name,prior_q[0],prior_q[1],prior_q[2],prior_q[3],prior_t[0],prior_t[1],prior_t[2],data_blob,
            ),
        )

    def add_poi(self, text_content,coordinate,sign_id,top_k_sign_id,poi_id = None):
        #检查list有没有转换为字符串
        if type(coordinate) == list:
            coordinate_store = json.dumps(coordinate,ensure_ascii=False)
        if type(text_content) == list:
            text_store = json.dumps(text_content,ensure_ascii=False)
        if type(sign_id) == list:
            sign_id_store = json.dumps(sign_id,ensure_ascii=False)
        if type(top_k_sign_id) == list:
            top_k_store = json.dumps(top_k_sign_id,ensure_ascii=False)
        cursor = self.execute(
            "INSERT INTO pois VALUES (?, ?, ?, ?, ?)",
            (poi_id,text_store,coordinate_store,sign_id_store,top_k_store))




if __name__ == '__main__':
    args = config.parse_args()
    group_path = args.group_path
    groups = os.listdir(group_path)
    json_path = args.shopsign_json #读取signboard 目标检测框和OCR结果处

    # 修改后
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        sign_data = data['sign']  # 改为读取sign list 
    for group_id in groups:
        database_path = r"{}/{}/colmap/database.db".format(group_path,group_id)
        sign_path = r"{}/{}/{}".format(group_path,group_id,args.sign_path) #sign.db
        points3D_path = r"{}/{}/colmap/aligned/points3D.txt".format(group_path,group_id)
        images_path = r"{}/{}/colmap/aligned/images.txt".format(group_path,group_id)
        #未能成功重建
        if not os.path.exists(points3D_path):
            continue
        if os.path.exists(sign_path):
            continue
            #os.remove(sign_path)
        db = SIGNDatabase.connect(sign_path)
        db.create_tables()
        print(group_id)

        # 读取images.txt中的每一行，与对应image的目标检测结果_dete.txt,ocr.txt比对，创建招牌对象
        keypoints_for_all = {}
        SignDete_for_all = {}
        image_names = []
        correspond_struct_idx = {}
        name_id = {}  # 存储IMAGE_NAME和IMAGE_ID的对应关系，
        
        #解析images.txt，提取图像ID、位姿、关键点及其关联的3D点索引

        with open(images_path, 'r') as file:
            skip_next = False  #控制是否跳过下一行的标志
            for index, line in enumerate(file):
                elements = line.split()
                if skip_next:
                    skip_next = False
                    continue
                correspond_struct_idx0 = []
                # 读取相机位姿并存储，便于坐标转换 [id, QW, QX, QY, QZ, TX, TY, TZ,NAME]
                if elements[0] == '#':
                    continue  # 直接跳过注释行
                if (index % 2 == 0):
                    image_id = int(elements[0])
                    try:
                        prior_q = np.array([float(element) for element in elements[1:5]])
                        prior_t = np.array([float(element) for element in elements[5:8]])
                        image_name = elements[-1] #1_front
                        image_name = image_name.split('.')[0]
                        name_id[image_name] = image_id
                    except ValueError:
                        skip_next = True  # 设置跳过下一行
                        continue  # 跳过当前行的剩余部分
                else:
                    key_points = []
                    for i, element in enumerate(elements):
                        if (i % 3 == 0):  # kp.pt.x
                            kp = []  # 不要在外部定义kp，会导致浅拷贝
                            kp.append(float(element))
                        elif (i % 3 == 1):  # kp.pt.y
                            kp.append(np.array(float(element)))
                            key_points.append(kp)
                        else:
                            correspond_struct_idx0.append(int(element))
                    key_points = np.asarray(key_points, np.float32)
                    db.add_image(image_id, image_name, correspond_struct_idx0, prior_q, prior_t)
                    db.add_keypoints(image_name, key_points)
                    correspond_struct_idx[image_name] = correspond_struct_idx0
                    keypoints_for_all[image_name] = key_points

        for image_name,key_points in keypoints_for_all.items():
            signs = [sign for sign in sign_data if Path(sign['image_name']).stem == image_name]  # 匹配去除扩展名的images
            #signs = {key: value for key, value in data.items() if key.startswith(image_name)}
            if signs != {}:
                for sign_id,value in signs.items():
                    text = ' '.join([text_entry['text'] for text_entry in value['texts']])
                    text_confidence = 0
                    text_confidence = sum(item['text_confidence'] for item in value['texts'])
                    text_confidence = text_confidence / len(value['texts'])
                    x_min, y_min, x_max, y_max = data[sign_id]["board_contour"]
                    point3d_idx = []
                    for i, kp in enumerate(key_points):
                        if (correspond_struct_idx[image_name][i] != -1 and x_min <= kp[1] <= x_max and y_min <= kp[
                            0] <= y_max):
                            point3d_idx.append(correspond_struct_idx[image_name][i])
                    db.add_sign(sign_id,x_min, y_min, x_max, y_max, point3d_idx,
                                     text,text_confidence,image_name)# def add_sign(self,sign_id,x_min,y_min,x_max,y_max,correspond_points3d_idx = None,text_content = None,text_confidence = None):

        #读取matches.txt
        with open(r"{}\{}\matches.txt".format(group_path,group_id), 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split(':')
                matches = parts[0].split(' ')
                matches = [image_name.split('.')[0] for image_name in matches]
                if parts[1] == '':  # 没有匹配结果
                    continue
                match_idx = list(map(int, parts[1].split(' ')))
                match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
                keypoints1 = keypoints_for_all.get(matches[0], None)
                keypoints2 = keypoints_for_all.get(matches[1], None)
                correspond_struct_idx1 = correspond_struct_idx.get(matches[0], None)
                correspond_struct_idx2 = correspond_struct_idx.get(matches[1], None)

                data1 = {key: value for key, value in data.items() if key.startswith(matches[0])}
                data2 = {key: value for key, value in data.items() if key.startswith(matches[1])}
                #feature_matches, features_img1, features_img2, data1, data2
                if not (data1 == {} or data2 == {}):
                    corresponding_ids,corresponding_idx = find_corresponding_sign(match_idx, keypoints1, keypoints2,data1, data2)
                    if corresponding_ids != []:
                        for i,corresponding_id in enumerate(corresponding_ids):
                            db.add_matches(corresponding_id[0],corresponding_id[1],corresponding_idx[i])

        points3d = {}
        idx_max = 0
        # 读取points3D.txt, points3d[idx] = [x,y,z]
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
        db.commit()
        db.close()