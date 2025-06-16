from gensim import corpora,models,similarities
import jieba
import os
from mycode.geocoding import gcj02_to_wgs84,wgs84_to_gcj02
from mycode.ecef2gps import gps2ecef
import numpy as np
from zhconv import convert
import config
import json
import csv
similarity_threshold = 0.5
spatial_threshold = 300

def find_max_similiarity(rows, match_idx):
    max_similiarity = 0
    min_row = None
    for row in rows:
        if row['match_idx'] == match_idx and float(row['similarity']) > max_similiarity:
            max_similiarity = float(row['similarity'])
            max_row = row
    return max_row


def read_json(input_path,outputpath):
    with open(input_path, 'r', encoding='utf-8') as file:
        poi_dict = json.load(file)
    column_headers = ['name','lat','lon','alt','match_POI','match_lat','match_lon','type','distance','similarity','match_idx','poi_id']
    with open(outputpath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_headers)
        # 写入列标题
        writer.writeheader()
        # 写入数据行
        for poi_id, poi_info in poi_dict.items():
            # 创建一个字典，包含所有需要的字段
            row_data = {
                'name': poi_info.get('name', ''),
                # 'coordinate': ','.join(map(str, poi_info.get('coordinate', []))),
                'lat':poi_info.get('coordinate')[0],
                'lon':poi_info.get('coordinate')[1],
                'alt':poi_info.get('coordinate')[2],
                'match_POI': poi_info.get('match_POI', ''),
                'match_coordinate': poi_info.get('match_coordinate', ''),  # 如果'match_coordinate'不存在，可以留空或设置默认值
                'type': poi_info.get('type', ''),
                'distance': poi_info.get('distance', ''),
                'similarity': poi_info.get('similarity', ''),
                'match_idx': poi_info.get('match_idx', ''),
                'poi_id': poi_id
            }

            if 'similarity' in row_data and row_data['similarity'] != '':
                # if float(row_data['similarity']) > 0.5 and row_data['distance']<spatial_threshold:
                if row_data['distance']<spatial_threshold:
                    if row_data['match_coordinate'] and isinstance(row_data['match_coordinate'], list):
                        row_data['match_lat'], row_data['match_lon'] = row_data['match_coordinate']
                    else:
                        row_data['match_lat'] = ''
                        row_data['match_lon'] = ''
                    del row_data['match_coordinate']
                # 写入数据行
                    writer.writerow(row_data)
def count_type(csv_path,key):
    # 创建一个字典来存储type字段的不同类型及其数量
    type_counts = {}

    # 打开CSV文件
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')  # 假设CSV文件使用制表符分隔

        # 遍历每一行
        for row in reader:
            # 获取type字段的值
            type_value = row[key]
            if key == '类别':
                parts = type_value.split(';')
                type_value = parts[0]
            # 如果type值已经在字典中，增加计数；否则，添加到字典并设置计数为1
            if type_value in type_counts:
                type_counts[type_value] += 1
            else:
                type_counts[type_value] = 1
    # 打印结果
    for type_value, count in type_counts.items():
        print(f"{type_value}: {count}")
    return type_counts


# def filter_poi(input_path,output_path):
#     with open(input_path, 'r',encoding='utf-8') as file:
#         reader = csv.DictReader(file, delimiter=',')
#         rows = list(reader)
#     # 处理数据
#     processed_rows = []
#     seen_match_idxs = set()
#
#     for row in rows:
#         match_idx = row['match_idx']
#         if match_idx not in seen_match_idxs:
#             processed_rows.append(row)
#             seen_match_idxs.add(match_idx)
#         else:
#             min_row = find_max_similiarity(processed_rows, match_idx)
#             if row['distance'] < min_row['distance']:
#                 processed_rows.remove(min_row)
#                 processed_rows.append(row)
#     # 写入新的CSV文件
#     fieldnames = rows[0].keys()
#     with open(output_path, 'w', newline='',encoding='utf-8-sig') as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
#         writer.writeheader()
#         writer.writerows(processed_rows)
def filter_poi(input_path,output_path):
    with open(input_path, 'r',encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        rows = list(reader)
    # 处理数据
    processed_rows = []
    seen_match_idxs = set()

    for row in rows:
        match_idx = row['match_idx']
        if match_idx not in seen_match_idxs:
            processed_rows.append(row)
            seen_match_idxs.add(match_idx)
        else:
            min_row = find_max_similiarity(processed_rows, match_idx)
            # if row['distance'] < min_row['distance']:
            if row['similarity'] > min_row['similarity']: #选择相似度最高的row
                processed_rows.remove(min_row)
                processed_rows.append(row)
    # 写入新的CSV文件
    fieldnames = rows[0].keys()
    with open(output_path, 'w', newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(processed_rows)

if __name__ == '__main__':
    matched_proportion = {}
    args = config.parse_args()
    poi_json_path = args.poi_output_path
    poi_csv_path = args.poi_csv_path
    matched_type = count_type("../../poi_before_filtered/matched_POI_filtered.csv",key = 'type')
    print("---------------------------------")
    ref_type = count_type("../../POI_ref/commercial_poi_wgs.csv",key = '类别')
    print("---------------------------------")
    for key in ref_type:
        if matched_type.get(key):
            matched_proportion[key] = matched_type[key]/ref_type[key]
        else:
            matched_proportion[key] = 0
    for type, proportion in matched_proportion.items():
        print(f"{type}: {proportion}")






