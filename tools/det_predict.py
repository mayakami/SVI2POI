import cv2
from ultralytics import YOLO
import os 
import gc
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import shutil
import json
from tqdm import tqdm
import config.det_config as config

def crop_and_save_detection(img, box, filename, idx, output_dir):
    """
    裁剪并保存目标检测框中的图像
    
    参数:
        img (numpy.ndarray): 原始图像
        box (numpy.ndarray): 边界框坐标 [x_min, y_min, x_max, y_max]
        image_name (str): 原始图像的文件名
        idx (int): 检测框索引
        output_dir (str): 裁剪图像保存目录
        
    返回:
        str: 裁剪图像的文件名
    """
    x_min, y_min, x_max, y_max = map(int, box)
    # 确保坐标在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img.shape[1], x_max)
    y_max = min(img.shape[0], y_max)
    
    # 裁剪图像
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    # 生成保存路径
    base_name = os.path.splitext(filename)[0]
    crop_filename = f"{base_name}_{idx}.jpg"
    output_path = os.path.join(output_dir, crop_filename)
    
    # 保存图像
    cv2.imwrite(output_path, cropped_img)
    
    return crop_filename

def save_to_json_format(boxes, classes, scores, crop_filenames, output_path):
    """将检测结果保存为JSON格式。
    
    参数:
        boxes (numpy.ndarray): 边界框坐标数组
        classes (numpy.ndarray): 类别索引数组
        scores (numpy.ndarray): 置信度分数数组
        crop_filenames (list): 裁剪图像文件名列表
        output_path (str): JSON文件保存路径
        
    示例:
        生成的JSON格式如下:
        {
            "image_name_0": {
                "board_contour": [477, 423, 657, 491],
                "confidence": 0.8488342761993408,
                "class": 0
            }
        }
    """
    result_dict = {}
    
    for i, (box, cls, score, crop_filename) in enumerate(zip(boxes, classes, scores, crop_filenames)):
        x_min, y_min, x_max, y_max = map(int, box)
        key = os.path.splitext(crop_filename)[0]  # 使用切割图像的文件名(不含扩展名)作为key
        
        result_dict[key] = {
            "board_contour": [
                x_min,
                y_min,
                x_max,
                y_max
            ],
            "confidence": float(score)
        }
    
    # 写入JSON文件
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)


def predict_signboard_json(
    args
):
    #dir_path = '/home/moss/streetview_segment/dataset/HongKong_download/demo'
    """
    使用YOLO模型检测图像中的招牌并保存检测结果
    
    参数:
        args: 包含以下字段的参数对象
            - model_path (str): YOLO模型的路径
            - dir_path (str): 基础目录路径
            - img_path (str): 相对于dir_path的输入图像路径或目录
            - output_path (str): 相对于dir_path的输出目录
            - save_crops (bool): 是否保存裁剪的招牌图像
            - iou_threshold (float): 非极大值抑制(NMS)的IoU阈值
            - conf_threshold (float): 目标检测的置信度阈值
            - batch_size (int): 批处理大小，用于加速推理
    """
    model_path = args.model_path
    dir_path = args.dir_path
    img_dir = os.path.join(dir_path, args.img_path)
    output_dir = os.path.join(dir_path, args.output_path)
    save_crops = args.save_crops
    iou_threshold = args.iou_threshold
    conf_threshold = args.conf_threshold
    batch_size = args.batch_size  # 批处理大小

    os.makedirs(output_dir, exist_ok=True)
    # 创建裁剪图像保存目录
    signboard_cut_dir = os.path.join(output_dir, 'signboard_cut')
    if save_crops:
        os.makedirs(signboard_cut_dir, exist_ok=True)

    # 获取待处理图像列表
    if os.path.isdir(img_dir):
        img_files = [f for f in os.listdir(img_dir) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')) 
                    and os.path.isfile(os.path.join(img_dir, f))]
        img_paths = [os.path.join(img_dir, f) for f in img_files]
    else:
        # 单个图像处理
        img_paths = [img_dir]
    if not img_paths:
        print("没有找到图像")
        return 0

    #加载模型
    model = YOLO(model_path)
    
    # 进行预测
    results = model.predict(
        source=img_paths, 
        iou=iou_threshold, 
        conf=conf_threshold, 
        save=False,
        stream=True,
        batch=batch_size
    )

    # 处理每个predict结果
    for result in tqdm(results, desc="处理图像", total=len(img_paths)):
        try:
            # 获取图像路径和文件名
            image_path = result.path
            image_name = os.path.basename(image_path)
            image_base = os.path.splitext(image_name)[0]
            json_output_path = os.path.join(output_dir, f"{image_base}.json")

            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像: {image_path}")
                continue
            image_height, image_width, _ = img.shape

            # 获取检测结果
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            scores = result.boxes.conf.cpu().numpy()  # 置信度分数
            classes = result.boxes.cls.cpu().numpy()  # 类别索引
            class_names = [model.names[int(cls)] for cls in classes]

            # 为每个检测框生成文件名（无论是否保存）
            crop_filenames = []
            for i in range(len(boxes)):
                crop_filename = f"{image_base}_{i}.jpg" # 使用图像基本名和检测框索引生成裁剪文件名
                crop_filenames.append(crop_filename)

            if save_crops and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    crop_and_save_detection(img, box, image_name, i, signboard_cut_dir)

            save_to_json_format(boxes, classes, scores, crop_filenames, json_output_path)
        except Exception as e:
            print(f"处理图像 {result.path} 时出错: {e}")
            # 释放内存
            del result
            gc.collect()

if __name__ == "__main__":
    args = config.parse_args()

    predict_signboard_json(args)