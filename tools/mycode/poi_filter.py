import json
import argparse
import sys
import os
import config.config_signboard_entity as config
import config.config_signboard_entity_area3 as area_config

def is_point_in_bbox(coordinate, bbox):
    """检查点是否在边界框内"""
    lat, lon = coordinate[0], coordinate[1]
    lat_min, lon_min, lat_max, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def filter_points_in_bbox(args,area_args):
    #input_file, output_file=None, bbox=None
    """筛选在边界框内的点"""
    input_file = args.prediction_json
    output_file = area_args.prediction_json
    bbox = area_args.bbox  # 从配置文件获取边界框坐标
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 筛选在边界框内的点
    points_in_bbox = {}
    
    for key, item in data.items():
        if "coordinate" in item and len(item["coordinate"]) >= 2:
            if is_point_in_bbox(item["coordinate"], bbox):
                points_in_bbox[key] = item
    
    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(points_in_bbox, f, ensure_ascii=False, indent=4)
    
    print(f"总点数: {len(data)}")
    print(f"边界框内点数: {len(points_in_bbox)}")
    
    return points_in_bbox

if __name__ == "__main__":
    args = config.parse_args()
    area_args = area_config.parse_args()
    filter_points_in_bbox(args,area_args)