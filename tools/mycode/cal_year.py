import os
import csv
from ecef2gps import gps2ecef
import numpy as np
import config
import warnings
warnings.filterwarnings('ignore')
#排除年份不对的图片
year_threshold = 2021 #2021年以前的GSV不能要
def read_imgcsv(path):
    image_gps = {}
    with open(path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用DictReader读取器，每行为一个字典
        for row in reader:
            image_name = row['name']
            image_gps[image_name] = {
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'alt': float(row['alt']),
                'rot': float(row['rot']),
                'date': int(row['date'])
            }
    return image_gps

def check_exist(group_dir):
    files_to_check = ['matches.txt', 'keypoints.txt', 'gsv_pairs.txt']
    all_files_exist = True

    for file_name in files_to_check:
        file_path = os.path.join(group_dir, file_name)
        if not os.path.exists(file_path):
            all_files_exist = False
            return all_files_exist
    if all_files_exist:
        return True

def change_image_gps(group_dir,image_gps):
    image_gps_sub = {}
    with open(os.path.join(group_dir, 'image_gps.csv'), mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用DictReader读取器，每行为一个字典
        for row in reader:
            image_name = row['name']
            if image_gps.get(image_name):
                if int(image_gps[image_name]['date']) < year_threshold:
                    continue
                else:
                    image_gps_sub[image_name] = image_gps[image_name]
    with open(os.path.join(group_dir, 'image_gps.csv'), 'w', newline='',
              encoding='utf-8-sig') as file:  # 写入image_gps.csv
        writer = csv.writer(file)
        writer.writerow(coordinate_w)  # 写入表头
        for key, value in image_gps_sub.items():
            writer.writerow([key, value['lat'], value['lon'], value['alt'], value['rot'], value['date']])
    return image_gps_sub

def change_group_pairs(group_dir,image_gps_sub,group_pairs_txt):
    group_pairs = []
    with open(os.path.join(group_dir, group_pairs_txt), 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            match = line.split(' ')
            if image_gps_sub.get(match[0]) and image_gps_sub.get(match[1]):
                group_pairs.append(line)

    with open(os.path.join(group_dir, group_pairs_txt), 'w', encoding='utf-8-sig') as file:
        for i in group_pairs:
            file.write(i + "\n")
    return group_pairs

def change_keypoints(group_dir,keypoints_txt):
    keypoints = []
    with open(os.path.join(group_dir, keypoints_txt), 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            if image_gps_sub.get(parts[0]):
                keypoints.append(line)

    with open(os.path.join(group_dir, keypoints_txt), 'w', encoding='utf-8-sig') as file:
        for i in keypoints:
            file.write(i + "\n")
    return keypoints

def change_matches(group_dir,matches_txt,group_pairs):
    matches = []
    with open(os.path.join(group_dir, matches_txt), 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            if parts[0] in group_pairs:
                matches.append(line)
    with open(os.path.join(group_dir, matches_txt), 'w', encoding='utf-8-sig') as file:
        for i in matches:
            file.write(i + "\n")
    return  matches
def write_colmap_gps(group_dir,colmap_gps_txt,image_gps_sub):
    colmap_gps = []
    with open(os.path.join(group_dir, colmap_gps_txt), 'w', encoding='utf-8-sig') as file:
        for key,value in image_gps_sub.items():
            line = f"{key} {value['lat']} {value['lon']} {value['alt']}"
            colmap_gps.append(line)
            file.write(line + "\n")
    return colmap_gps

if __name__ == "__main__":
    args = config.parse_args()
    group_path = args.group_path
    groups = os.listdir(group_path)
    img_csv = args.img_csv
    group_pairs_txt = args.group_pairs_txt
    image_gps = read_imgcsv(img_csv) #读取img_csv
    coordinate_w = ['name','lat','lon','alt','rot','date']
    for i in groups:
        group_dir = os.path.join(group_path, i)
        if not check_exist(group_dir):
            continue
        image_gps_sub = change_image_gps(group_dir,image_gps)
        group_pairs = change_group_pairs(group_dir,image_gps_sub,group_pairs_txt)
        keypoints = change_keypoints(group_dir,'keypoints.txt')
        matches = change_matches(group_dir,'matches.txt',group_pairs)
        colmap_gps = write_colmap_gps(group_dir,'colmap/image_gps.txt',image_gps_sub)






