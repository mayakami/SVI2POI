import os
import argparse
import time
from datetime import datetime

# 导入各个组件
from det_predict import predict_signboard_json

from ocr_predict import scene_ocr_predict
from poi_cluster import create_signs,create_poi,save_poi

from poi_processor import llm_poi
from chatglm_predict import run_evaluation
from read_predict import read_predict

# 导入配置
import config.config_signboard_entity as config
import config.det_config as det_config
import config.ocr_config as ocr_config

if __name__ == "__main__":
    det_args = det_config.parse_args()
    ocr_args = ocr_config.parse_args()
    args = config.parse_args()
    predict_signboard_json(det_args)
    scene_ocr_predict(ocr_args)
    group_path = args.group_path
    groups = os.listdir(group_path) 
    for i in groups:
        create_signs(args, i)#创建并聚类招牌示例
        POIs = create_poi(args,i) 
        save_poi(args, POIs, i)
    llm_poi(args)
    run_evaluation(args)
    read_predict(args)




