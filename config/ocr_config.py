
import argparse
import json

ROI_LABEL_MAP = {
    "Signboard": 0,
    "Billboard": 1,
    "Streetsign": 2,
    "Others": 3
}

TEXT_LABEL_MAP = {
    "Prefix": 0,
    "Title": 1,
    "Subtitle": 2,
    "Address": 3,
    "Tag": 4,
    "Tel": 5,
    "Others": 6
}


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0) 

    # paras for filepath
    parser.add_argument("--img_dir", type=str,
                        default='/home/moss/streetview_segment/dataset/HongKong_download/gsv_cut')
    parser.add_argument("--input_json", type=str, help= "从JSON文件加载已有的招牌检测/标注结果",
                    default='/home/moss/github_subject/gsv_label/signboard entity/area3/signboard entity.json')
    parser.add_argument("--output_json", type=str, help= "输出json文件路径(加了一个ocr_result字段)",
                default='/home/moss/github_subject/gsv_label/signboard entity/area3/signboard_entity_ocr.json')
    parser.add_argument("--save_dirpath", type=str, help= "从JSON文件加载已有的招牌检测/标注结果",
                    default='/home/moss/github_subject/gsv_label/signboard entity/area3/')
    parser.add_argument("--save_imgpath", type=str,
                    default='/home/moss/streetview_segment/dataset/HongKong_download/dect_img')

    # params for ROI detection
    parser.add_argument("--backbone", type=str, default='swin')
    parser.add_argument("--model_type", type=str, default='cascade_mask_rcnn')
    parser.add_argument("--config_file", type=str,
                        default='../output/cascade_mask_rcnn_swin_cls2/shopsign_cascade_mask_rcnn_swin.py')
    parser.add_argument("--checkpoint_file", type=str, default='../output/cascade_mask_rcnn_swin_cls2/latest.pth')
    parser.add_argument("--roi_threshold", type=float, default=0.5)

    #params for PaddleOCR
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")
    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    parser.add_argument("--onnx_providers", nargs="+", type=str, default=False)
    parser.add_argument("--onnx_sess_options", type=list, default=False)
    parser.add_argument("--det_box_type", type=str, default="quad")
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--use_mlu", type=str2bool, default=False)
    parser.add_argument(
        "--use_gcu",
        type=str2bool,
        default=False,
        help="Use Enflame GCU(General Compute Unit)",
    )
    parser.add_argument(
        "--return_word_box",
        type=str2bool,
        default=False,
        help="Whether return the bbox of each word (split by space) or chinese character. Only used in ppstructure for layout recovery",
    )
    parser.add_argument("--save_crop_res", type=str2bool, default=False)

    # params for text detector
    parser.add_argument("--det_algorithm", type=str, default='DB') 
    parser.add_argument("--det_model_dir", type=str, default='/home/moss/streetview_segment/tools/PaddleOCR/model/det/ch_PP-OCRv4_det_server_infer/')
    #parser.add_argument("--det_model_dir", type=str, default='../output/det_r101_vd_db_ch/infer')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="slow")
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str, default='/home/moss/streetview_segment/tools/PaddleOCR/model/rec/ch_PP-OCRv4_rec_server_infer/')
    #parser.add_argument("--rec_image_shape", type=str, default="3, 64, 256")
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=1)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="/home/moss/streetview_segment/tools/PaddleOCR/model/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt",
        help="Path to the character dictionary used for recognition.")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="/home/moss/streetview_segment/tools/PaddleOCR/model/PaddleOCR/doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.3)

    parser.add_argument("--use_angle_cls", type=str2bool, default=False)

    # paras for ROI and text line classification
    parser.add_argument("--vlcls_checkpoint_file", type=str, default='../output/vl_cls_all_params/net.pkl')
    parser.add_argument("--roi_label_map", type=str, default=json.dumps(ROI_LABEL_MAP))
    parser.add_argument("--text_label_map", type=str, default=json.dumps(TEXT_LABEL_MAP))

    return parser.parse_args()
