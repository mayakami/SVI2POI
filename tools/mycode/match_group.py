import os
import csv
from ecef2gps import gps2ecef
import numpy as np
import config
import warnings
warnings.filterwarnings('ignore')
#生成图像匹配对txt文件，并保存在各自的文件夹中
#在exhuastive matching的基础上加入距离约束，超过30m的图像就不加入了
#有改动,img_gps.csv中加了旋转
max_distance = 30 #图像匹配距离阈值
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
coordinate_w = ['name', 'lat', 'lon', 'alt', 'rot']

#chunk_size:每组图片数量，overlap_size：重叠图片数量
def chunk_list_with_overlap(full_list, chunk_size, overlap_size):
    chunked_list = []
    i = 0  # 起始索引
    while i < len(full_list):
        # 根据重叠度计算每次增加的步长
        # 从列表中切片
        chunk = full_list[i:i + chunk_size]
        i += chunk_size - overlap_size
        # 将切片添加到结果列表
        chunked_list.append(chunk)
        # 如果到达列表末尾，则停止
        if i + chunk_size >= len(full_list):
            break
    return chunked_list
if __name__ == "__main__":
    #参数
    args =config.parse_args()
    img_path = args.img_path
    group_path = args.group_path
    ouptut_txt = args.group_pairs_txt
    ouptut_csv = args.group_gps_csv
    group_ref = "C:/Users/24745/Desktop/test/points_divided3.csv"#arcgis生成的point分组结果
    #每个group中至少要包含多少张图片才能保证重建质量？
    group_dict = {} #{'group_id':['name',lat,lon,alt,rot]}
    with open(group_ref, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用DictReader读取器，每行为一个字典
        for row in reader:
            # 提取group作为字典键
            group_key = row['group']
            values = [row['name'], float(row['lat']), float(row['lon']), float(row['alt']), float(row['rot'])]
            # 将列表添加到对应group的键下
            if group_dict.get(group_key):
                group_dict[group_key].append(values)
            else:
                group_dict[group_key] = [values]

    for group_id, value in group_dict.items():
        if len(value)>10:
            max_distance = 30
        else:
            max_distance = 50
        path_name = os.path.join(group_path,group_id)
        os.makedirs(path_name, exist_ok=True)
        img_csv = r"{}\{}\{}".format(group_path, group_id, args.img_csv) #需要重新写入img_csv，这样子的话read_metadata只需要生成arcgis所需的前置文件
        img_names = []
        data_dict = {}
        img_group = []
        with open(img_csv, 'w', newline='', encoding='utf-8-sig') as file:  # 写入image_gps.csv
            writer = csv.writer(file)
            writer.writerow(coordinate_w)  # 写入表头
            for item in value:
                image_id = item[0]
                for suffix, offset in faceTransform.items():
                    new_name = f"{image_id}_{suffix}.jpg"
                    new_rot = item[-1] + offset  # 计算新的rot值
                    # 写入新行
                    writer.writerow([new_name, item[1], item[2], item[3], new_rot])
                    img_group.append([new_name, item[1], item[2], item[3], new_rot])

        with open(os.path.join(path_name,ouptut_txt),'w',encoding='utf-8-sig') as file:
            for i in range(len(img_group)):
                for j in range(i + 1, len(img_group)):
                    distance = np.linalg.norm( (img_group[i][1:4])-gps2ecef(img_group[j][1:4]))
                    if distance > max_distance:
                        continue
                    else:
                        pair = f"{img_group[i][0]} {img_group[j][0]}"
                        file.write(pair + "\n")
    print("match group done")














    # # img_csv = args.img_csv
    # groups = os.listdir(group_path)
    # for group_id in groups:
    #     img_csv = r"{}\{}\{}".format(group_path,group_id,args.img_csv)
    #     img_names = []
    #     data_dict = {}
    #     coordinate_w = ['name', 'lat', 'lon', 'alt', 'rot']
    #     with open(img_csv, mode='r', encoding='utf-8-sig') as csvfile:
    #         reader = csv.reader(csvfile)
    #         headers = next(reader)  # 列标题
    #         for row in reader:
    #             img_names.append(row[0].strip())
    #             data_dict[row[0]] = [float(item) for item in row[1:]]
    #     print("#")


    # #
    # img_names = []
    # data_dict = {}
    # coordinate_w = ['name', 'lat', 'lon', 'alt', 'rot']
    #
    # #读取存储所有gps信息的csv文件，存入data_dict字典中
    # with open(img_csv, mode='r',encoding='utf-8-sig') as csvfile:
    #     reader = csv.reader(csvfile)
    #     headers = next(reader) #列标题
    #     for row in reader:
    #         img_names.append(row[0].strip())
    #         data_dict[row[0]] = [float(item) for item in row[1:]]
    #
    # # 计算图片索引分组 可能真的要修改了
    # chunked_images = chunk_list_with_overlap(img_names, 96, 16)
    # for idx,img_group in enumerate(chunked_images):
    #     path_name = os.path.join(group_path,str(idx))
    #     os.makedirs(path_name, exist_ok=True)
    #     with open(os.path.join(path_name,ouptut_csv),'w',newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(coordinate_w)
    #         for img in img_group:
    #             lat, lon, alt, rot = data_dict[img]
    #             writer.writerow([img, lat, lon, alt, rot])
    #     with open(os.path.join(path_name,ouptut_txt),'w',encoding='utf-8-sig') as file:
    #         for i in range(len(img_group)):
    #             for j in range(i + 1, len(img_group)):
    #                 distance = np.linalg.norm(gps2ecef(data_dict[img_group[i]][0:3]) - gps2ecef(data_dict[img_group[j]][0:3]))
    #                 if distance > max_distance:
    #                     continue
    #                 else:
    #                     pair = f"{img_group[i]} {img_group[j]}"
    #                     file.write(pair + "\n")



    # image_path = 'F:\data\HongKong_download\test'
    # ouptut_path = "..\..\group" #分组存储
    # ouptut_txt = "gsv_pairs.txt"
    # ouptut_csv = "image_gps.csv"
    # img_names = []
    # data_dict = {}
    # coordinate_w = ['name', 'lat','lon', 'alt','rot']
    # #读取存储所有gps信息的csv文件，存入data_dict字典中
    # with open("..\..\image_gps.csv", mode='r',encoding='utf-8-sig') as csvfile:
    #     reader = csv.reader(csvfile)
    #     headers = next(reader) #列标题
    #     for row in reader:
    #         img_names.append(row[0].strip())
    #         data_dict[row[0]] = [float(item) for item in row[1:]]
    #
    # #计算图片索引分组
    # chunked_images = chunk_list_with_overlap(img_names, 96, 16)
    #
    # for idx,img_group in enumerate(chunked_images):
    #     path_name = os.path.join(ouptut_path,str(idx))
    #     os.makedirs(path_name, exist_ok=True)
    #     with open(os.path.join(path_name,ouptut_csv),'w',newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(coordinate_w)
    #         for img in img_group:
    #             lat, lon, alt, rot = data_dict[img]
    #             writer.writerow([img, lat, lon, alt, rot])
    #     with open(os.path.join(path_name,ouptut_txt),'w',encoding='utf-8-sig') as file:
    #         for i in range(len(img_group)):
    #             for j in range(i + 1, len(img_group)):
    #                 distance = np.linalg.norm(gps2ecef(data_dict[img_group[i]][0:3]) - gps2ecef(data_dict[img_group[j]][0:3]))
    #                 if distance > max_distance:
    #                     continue
    #                 else:
    #                     pair = f"{img_group[i]} {img_group[j]}"
    #                     file.write(pair + "\n")
    # print("match group done")



