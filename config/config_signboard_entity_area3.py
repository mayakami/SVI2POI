import argparse
import json

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    #database
    parser.add_argument("--streetview_path", type=str,
                        default="/home/moss/streetview_segment/")
    parser.add_argument("--group_path", type=str,
                    default="/home/moss/streetview_segment/group")
    parser.add_argument("--img_path", type=str,
                        default="/home/moss/streetview_segment/dataset/HongKong_download/gsv_cut")
    parser.add_argument("--sign_path", type=str,
                    default="sign_area3.db")
    parser.add_argument("--metadata_path", type=str,
                        default="/home/moss/streetview_segment/dataset/HongKong_download/metadata")
    parser.add_argument("--dect_path", type=str,
                        default="/home/moss/streetview_segment/dataset/HongKong_download/signboard_detect")
    parser.add_argument("--shopsign_json",type=str,
                        default = "/home/moss/streetview_segment/dataset/gsv_label/signboard_entity/area3/signboard_entity.json")
    parser.add_argument("--poi_json",type=str,
                        default = "poi_area3.json")
    parser.add_argument("--error_matching_dir",type=str,
                        default = "/home/moss/streetview_segment/tools/mycode/match_results/area3/error_matching")
    parser.add_argument("--merged_poi_json",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/merged_poi.json")
    parser.add_argument("--dev_json",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/dev.json")
    parser.add_argument("--prediction_txt",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/generated_predictions.txt")
    parser.add_argument("--prediction_json",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/merged_poi_with_name.json")
    parser.add_argument("--match_json",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/poi_match.json")
    parser.add_argument("--mismatch_path",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/poi_mismatch.txt")      
    parser.add_argument("--distance_result_dir",type=str,
                        default = "/home/moss/streetview_segment/poi_dataset/area3/distance_result")
    parser.add_argument("--match_path", type=str,
                        default="/home/moss/streetview_segment/poi_dataset/area3/poi_match.json")
    parser.add_argument("--recall_path", type=str,
                        default="/home/moss/streetview_segment/poi_dataset/area3/poi_recall.json")
    #parser for metadata
    parser.add_argument("--metadata_csv", type=str,
                        default="../../document/metadata_gps.csv")
    parser.add_argument("--img_csv", type=str,
                        default="../../document/image_gps.csv")
    # parser for poi download
    parser.add_argument("--gaodeKey", type=str,
                        default="5ec21e666018abf98f2a766553b2f240")
    parser.add_argument("--poi_type", type=str,
                        default='汽车服务|生活服务|汽车销售|汽车维修|摩托车服务|餐饮服务|购物服务|'\
                '生活服务|体育休闲服务|医疗保健服务|住宿服务|风景名胜|商务住宅|'\
                '政府机构及社会团体|科教文化服务|交通设施服务|金融保险服务|公司企业|'\
                '道路附属设施|地名地址信息|公共设施|事件活动|虚拟数据|通行设施')
    parser.add_argument("--polyline_path", type=str,
                        default="F:/data/HongKong_road_network/output/residential_polyline_gcj.json")

    #parser for image group
    parser.add_argument("--group_pairs_txt", type=str,
                        default="gsv_pairs.txt")
    parser.add_argument("--group_gps_csv", type=str,
                        default="image_gps.csv")
    parser.add_argument("--img_size", type=str, default="1024,1024")
    parser.add_argument("--camera_intrinsic", type=str, default="1228,1228,512,512")
    # parser for colmap
    parser.add_argument("--ref_txt", type=str, default= "image_gps.txt")
    parser.add_argument("--poi_ref_type", type=str,
                        default="wgs")
    parser.add_argument("--poi_output_path", type=str, default="../../poi_all.json")
    parser.add_argument("--poi_csv_path", type=str, default="../../matched_poi.csv")
    parser.add_argument("--bbox", type=float, nargs=4,
                default=[22.301212, 114.171769, 22.303588, 114.176583],
                help="边界框坐标: lat_min lon_min lat_max lon_max")

    return parser.parse_args()