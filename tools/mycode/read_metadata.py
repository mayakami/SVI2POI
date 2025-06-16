import json
from json.decoder import JSONDecodeError
import os
import re
import csv
import config
#读取metadata中的经纬度坐标，并存入csv文件中'./image_gps.csv'
def sort_by_numeric_order(file):
    # base_name = os.path.splitext(file)[0].split('_',',')[0]  # 获取不带扩展名的文件名
    base_name = re.split('[_.]', os.path.splitext(file)[0])[0]
    number = int(base_name)  # 将文件名转换为整数
    return number


faceTransform = {
    "front":0,
    "right": 90,
    "back": 180,
    "left": 270,
    "rightfront": 45,
    "rightback": 135,
    "leftback": 225,
    "leftfront": 315
}

def write_to_csv(filename, headers, rows):
    """写入数据到 CSV 文件。
    Args:
        filename (str): CSV 文件的完整路径。
        headers (list): CSV 文件的头部标题。
        rows (list of lists): 要写入的行数据。
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == "__main__":
    #从metadata数据集中读取GPS坐标
    #目标格式:
    #image_name1.jpg lat lon alt
    #image_name2.jpg lat lon alt

    args = config.parse_args()
    metadata_csv = args.metadata_csv
    img_csv = args.img_csv
    metadata_path = args.metadata_path
    img_path = args.img_path
    coordinate_w = ['name', 'lat', 'lon', 'alt', 'rot','date']

    #读取文件夹中所有文件名并排序
    file_ls = os.listdir(metadata_path)
    image_ls = os.listdir(img_path)
    file_ls = sorted(file_ls, key=sort_by_numeric_order)
    image_ls = sorted(image_ls, key=sort_by_numeric_order)

    # 读取GSV的metadata "F:\data\streetview_test\metadata\*.metadata.json",
    gps = {}  # {image_id:[lon,lat,alt]}
    rows = []
    date = {}
    for src_file in file_ls:
        try:
            with open(os.path.join(metadata_path,src_file),'r',encoding='utf-8-sig') as file:
                data = json.load(file)
            finalSaveLocation = data['finalSaveLocation'] #文件名
            image_name = os.path.basename(finalSaveLocation)
            image_id = int(image_name.split('.')[0])
            lat = data['lat']
            lon = data['lng']
            alt = data['elevation'] if 'elevation' in data and data['elevation'] else 6
            year = data['date']['year']
            rotation = data['rotation']
            gps[image_id] = [lat,lon,alt]
            date[image_id] = year
            rows.append([image_id, lat, lon, alt, rotation, year])
            #顺便把depthMapString删了
            #不能删啊QAQ,删了好像也没关系,已经是压缩后的了,但还是不要删吧
            # if 'depthMapString' in data:
            #     del data['depthMapString']
            # if 'thumbnailString' in data:
            #     del data['thumbnailString']
            with open(os.path.join(metadata_path,src_file), 'w') as file:
                json.dump(data,file,indent=4)
        except JSONDecodeError as e:
            print("JSONDecodeError occurred: {0}. Skipping file: {1}".format(e, src_file))
        except Exception as e:
            print("An error occurred: {0}. Skipping file: {1}".format(e, src_file))

    write_to_csv(metadata_csv,coordinate_w,rows)  # 写入metadata_gps.csv

    with open(img_csv, 'w', newline='', encoding='utf-8-sig') as file:  # 写入image_gps.csv
        writer = csv.writer(file)
        writer.writerow(coordinate_w) #写入表头
        for src_file in image_ls:
            image_id = int(src_file.split('_')[0])
            image_look = src_file.split('_')[1].split('.')[0]
            rotation_save = rotation + faceTransform[image_look]
            lat, lon, alt = gps[image_id]
            year = date[image_id]
            writer.writerow([src_file, lat, lon, alt, rotation_save, year])
    print("done")


