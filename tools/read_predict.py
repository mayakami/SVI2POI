import json
import config.config_signboard_entity as config

def read_predict(args):
    """
    从预测结果JSON文件中读取POI数据，将name字段添加到merged_poi.json中，并保存为新的JSON文件

    参数:
        prediction_json_path: 输入的预测结果JSON文件路径
        merged_poi_path: 原始merged_poi.json文件路径
        ref_json_path: 参考的ref_json文件路径
        output_json_path: 输出的JSON文件路径
    """
    prediction_txt_path = args.prediction_txt  # 预测结果文件路径
    merged_poi_path = args.merged_poi_json  
    ref_json_path = args.dev_json  # 参考的ref_json文件路径
    output_json_path = args.prediction_json  
    # 读取预测结果
    predictions = []
    with open(prediction_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                pred_item = json.loads(line)
                predictions.append(pred_item)
            except json.JSONDecodeError as e:
                print(f"解析错误: {e}")
                print(f"问题行内容: {line}")
    
    print(f"已读取 {len(predictions)} 条预测结果")

    # 读取merged_poi.json文件
    with open(merged_poi_path, 'r', encoding='utf-8') as f:
        merged_poi_data = json.load(f)
    
    print(f"已读取 {len(merged_poi_data)} 条原始POI数据")

    # 读取ref_json文件
    ref_data = []
    with open(ref_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                ref_item = json.loads(line)
                ref_data.append(ref_item)
            except json.JSONDecodeError as e:
                print(f"ref_json 解析错误: {e}")
                print(f"问题行内容: {line}")
    
    print(f"已读取 {len(ref_data)} 条参考数据")

    # 检查数据量是否一致
    if len(predictions) != len(ref_data):
        print(f"警告: 预测结果数量 ({len(predictions)}) 与参考数据数量 ({len(ref_data)}) 不一致")

    # 为每个POI添加name字段
    for i, (poi_id, poi_info) in enumerate(merged_poi_data.items()):
        if i < len(ref_data):
            # 获取对应的point_seq_id
            point_seq_id = ref_data[i].get("point_seq_id")
            if point_seq_id == poi_id:
                # 如果point_seq_id与poi_id匹配，则使用预测结果
                if i < len(predictions):
                    merged_poi_data[poi_id]["name"] = predictions[i].get("predict", "")
                else:
                    print(f"警告: POI ID {poi_id} 没有对应的预测结果")
                    merged_poi_data[poi_id]["name"] = "未知"
            else:
                print(f"警告: POI ID {poi_id} 与 point_seq_id {point_seq_id} 不匹配")
                merged_poi_data[poi_id]["name"] = "未知"
        else:
            print(f"警告: POI ID {poi_id} 没有对应的参考数据")
            merged_poi_data[poi_id]["name"] = "未知"
    
    # 保存为JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_poi_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成! 已保存到 {output_json_path}")

if __name__ == "__main__":
    args = config.parse_args()
    read_predict(args)