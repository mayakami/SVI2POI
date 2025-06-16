import argparse
import json
import os  # 添加 os 模块导入

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--streetview_path", type=str,
                        default="/home/moss/streetview_segment/")
    parser.add_argument("--dir_path", type=str,
                        default="/home/moss/streetview_segment/dataset/HongKong_download/")
    parser.add_argument("--model_path", type=str,
                        default='/home/moss/streetview_segment/tools/YOLO/ultralytics/runs/train/signboard_detect/best.pt',
                        help="Path to the model weights file")
    parser.add_argument("--img_path", type=str,
                        default="gsv_cut")
    parser.add_argument("--output_path", type=str,
                        default="/home/moss/streetview_segment/dataset/HongKong_download/signboard_detect",
                        help="Directory for detection output")
    parser.add_argument("--conf_threshold", type=float,
                        default=0.45,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou_threshold", type=float,
                        default=0.3,
                        help="IoU threshold for non-maximum suppression")
    parser.add_argument("--save_crops", type=str2bool,
                        default=True,
                        help="Whether to save cropped detection regions")
    parser.add_argument("--batch_size", type=float,
                        default=4)
    args = parser.parse_args()
    return args