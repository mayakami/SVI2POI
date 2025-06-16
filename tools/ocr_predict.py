import os
import sys
import mmcv
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json
import sys 
import config.ocr_config as config
import PaddleOCR.ocr_predict as ocr_predict
import PaddleOCR.predict_utils as predict_utils
#sys.path.insert(1, '/home/moss/github_subject/scene-image-poi-generation/')

def predict_scene_poi(args):
    """
    对招牌检测结果进行OCR文本识别并更新JSON文件
    
    参数:
        args: 配置参数对象，必须包含以下属性:
            detect_dir: 招牌检测结果JSON文件所在目录路径
            img_dir: 对应图片文件所在目录路径
    """
    ocr_recognizer = ocr_predict.OCRRecognizer(args)
    # 获取检测目录下所有JSON文件
    json_files = [f for f in os.listdir(args.detect_dir) if f.endswith('.json')]
    for json_file in tqdm(json_files, desc="处理文件"):
        # 读取检测结果JSON
        json_path = os.path.join(args.detect_dir, json_file)
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            detection_data = json.load(f)
    # 提取基本图片名称（移除数字后缀）
        base_name = json_file.replace('.json', '')
        img_name = f"{base_name}.jpg"
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.exists(img_path):
                print(f"警告: 图片 {img_path} 未找到，跳过...")
                continue
        # 加载图片
        img = mmcv.imread(img_path)
        # 处理每个检测框
        modified = False
        for key, info in detection_data.items():
            '''
            # 如果已经有OCR结果，则跳过
            if 'ocr_result' in info:
                continue
            '''
            # 获取招牌区域坐标
            board_contour = info.get('board_contour', [])
            if not board_contour or len(board_contour) != 4:
                continue
            # 解析坐标为x1,y1,x2,y2格式
            x1, y1, x2, y2 = map(int, board_contour)
            try:
                # 生成ROI区域
                roi_img = img[y1:y2, x1:x2]
                
                # 进行OCR识别
                ocr_result = ocr_recognizer(roi_img)
                
                # 将OCR结果格式转为仅文本列表
                texts = [text_info['text'] for text_info in ocr_result]
                
                # 添加OCR结果到原JSON
                info['ocr_result'] = texts
                modified = True
            except Exception as e:
                print(f"处理 {json_file} 中 {key} 时出错: {str(e)}")
        if modified:
            # 保存回原文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, ensure_ascii=False, indent=2)
            print(f"已更新文件: {json_file}")
    '''
    # 按图片名称分组处理
    img_groups = {} #{image_name:[signboard_entity]}
    for sign in data['sign']:
        img_name = sign['image_name']
        if img_name not in img_groups:
            img_groups[img_name] = []
        img_groups[img_name].append(sign)
    
    # 处理每个图片
    for img_name, signs in tqdm(img_groups.items()):
        img_path = os.path.join(args.img_dir, img_name)
        img = mmcv.imread(img_path)
        for sign in signs:
            # 获取招牌区域坐标
            board_contour = sign.get('board_contour', [])
                
            # 解析坐标为x1,y1,x2,y2格式
            x1, y1, x2, y2 = map(int, board_contour)
            
            # 生成ROI区域
            roi_img = img[y1:y2, x1:x2]
            
            # 进行OCR识别
            ocr_result = ocr_recognizer(roi_img)  #注意可能需要根据OCR模型调整输入
            
            # 将OCR结果保存到当前sign条目中
            sign['ocr_result'] = []
            for text_info in ocr_result:
                # 转换坐标系到原图
                adjusted_boxes = []
                for point in text_info['text_box']:
                    adjusted_x = point[0] + x1
                    adjusted_y = point[1] + y1
                    adjusted_boxes.append([adjusted_x, adjusted_y])
                
                sign['ocr_result'].append({
                    'text': text_info['text'],
                    #'score': text_info['score'],
                    #'text_box': adjusted_boxes,
                })
    
    # 保存到新的JSON文件
    output_path = os.path.join(args.save_dirpath, args.output_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    '''
if __name__ == "__main__":
    args = config.parse_args()
    # 需要确保以下参数在config中已正确配置：
    # args.input_json - 输入的包含招牌检测结果的JSON路径
    # args.img_dir - 图片存放的目录路径 
    # args.save_dirpath - 结果保存目录
     # 设置路径
    dir_path = '/home/moss/streetview_segment/dataset/HongKong_download/demo'
    args.img_dir = os.path.join(dir_path, 'gsv_cut')  # 图片目录
    args.detect_dir = os.path.join(dir_path, 'signboard_detect')  # 检测结果目录
    predict_scene_poi(args)