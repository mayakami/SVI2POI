import json
import os
import config_signboard_entity_area3 as config

sim_threshold = 0.15

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_metrics(tp, fp, fn):
    """计算准确率、召回率和F1值"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_poi_matches(best_matches_path, ground_truth_path, sim_threshold=0.6):
    """
    评估POI匹配的准确率、召回率和F1值，分别计算总体、GD和manual匹配指标
    
    参数:
        best_matches_path: best_poi_matches.json的路径
        ground_truth_path: signboard_entity_ocr.json的路径
        sim_threshold: 相似度阈值，只有超过该值的匹配才被视为正确
    
    返回:
        包含总体、GD和manual匹配指标的字典
    """
    # 加载文件
    best_matches = load_json_file(best_matches_path)
    
    # 分离manual和非manual(即GD)的匹配结果
    manual_poi_to_match = {}
    gd_poi_to_match = {}
    
    # 首先创建所有匹配的POI字典
    poi_to_match = {}
    for key in best_matches:
        if ("match_POI" in best_matches[key] and 
            "poi_id" in best_matches[key]["match_POI"]):
            match_poi = best_matches[key]["match_POI"]
            poi_id = match_poi["poi_id"]
            poi_to_match[poi_id] = match_poi
            
            # 按匹配类型分类
            if "type" in match_poi and match_poi["type"] == "manual":
                manual_poi_to_match[poi_id] = match_poi
            else:
                # 所有非manual类型都视为GD
                gd_poi_to_match[poi_id] = match_poi
    
    ground_truth_data = load_json_file(ground_truth_path)
    # 将ground truth也按照match_type分类
    all_ground_truth_pois = {poi["poi_id"]: poi for poi in ground_truth_data["poi"]}
    manual_ground_truth_pois = {poi["poi_id"]: poi for poi in ground_truth_data["poi"] 
                              if "match_type" in poi and poi["match_type"] == "manual"}
    gd_ground_truth_pois = {poi["poi_id"]: poi for poi in ground_truth_data["poi"] 
                          if "match_type" not in poi or poi["match_type"] != "manual"}
    # 计算各类别的指标
    total_metrics = calculate_type_metrics(poi_to_match, all_ground_truth_pois, sim_threshold)
    manual_metrics = calculate_type_metrics(manual_poi_to_match, manual_ground_truth_pois, sim_threshold)
    gd_metrics = calculate_type_metrics(gd_poi_to_match, gd_ground_truth_pois, sim_threshold)
    
    return {
        "total": total_metrics,
        "manual": manual_metrics,
        "gd": gd_metrics
    }

def calculate_type_metrics(poi_to_match, ground_truth_pois, sim_threshold):
    """
    计算特定类型(total/manual/gd)匹配的指标
    """
    # 初始化计数器
    tp, fp, fn = 0, 0, 0

    # 计算 tp 和 fp
    for poi_id, poi_info in poi_to_match.items():
        if poi_id in ground_truth_pois:
            # 检查相似度是否高于阈值
            if 'similarity' in poi_info and poi_info['similarity'] > sim_threshold:
                tp += 1  # 真阳性
            else:
                fp += 1  # 假阳性（相似度低）
        else:
            fp += 1  # 假阳性（不在参考数据中）

    # 计算 fn
    for gt_poi_id in ground_truth_pois:
        if gt_poi_id not in poi_to_match:
            fn += 1  # 假阴性

    # 计算评估指标
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def calculate_average_distance(json_file_path):
    """
    计算JSON文件中所有包含match_POI.distance字段且similarity>0.6的条目的平均距离值
    
    参数:
        json_file_path: JSON文件的路径
    
    返回:
        平均距离值、正确匹配的条目数量
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 收集所有符合条件的distance值
    distance_values = []
    
    # 遍历JSON中的每个条目
    for key, value in data.items():
        # 检查是否存在match_POI字段、similarity子字段和distance子字段，且similarity>0.6
        if ('match_POI' in value and 
            'similarity' in value['match_POI'] and 
            'distance' in value['match_POI'] and
            value['match_POI']['similarity'] > sim_threshold):
            
            distance = value['match_POI']['distance']
            distance_values.append(distance)
    
    # 计算平均值
    if distance_values:
        average_distance = sum(distance_values) / len(distance_values)
        return average_distance, len(distance_values)
    else:
        return 0, 0

def calculate_match_accuracy(match_json_path,predict_json_path):
    """
    计算正确匹配的生成POI数量与生成的POI总数量之比
    
    参数:
        match_json_path: 包含匹配结果的JSON文件路径
        
    返回:
        准确率, 正确匹配的POI数量, 总POI数量
    """
    try:
        # 加载匹配结果数据
        match_data = load_json_file(match_json_path)
        predict_data = load_json_file(predict_json_path)
        # 统计总POI数量
        total_pois = len(predict_data)
        
        # 统计有正确标签的POI数量
        correct_matches = 0
        for poi_id, poi_info in match_data.items():
            # 检查POI是否有标签字段并正确匹配
            if "match_POI" in poi_info :
                correct_matches += 1
        
        # 计算准确率
        accuracy = correct_matches / total_pois if total_pois > 0 else 0
        
        return accuracy, correct_matches, total_pois
    
    except Exception as e:
        print(f"计算匹配准确率时出错: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # 文件路径
    args = config.parse_args()
    prediction_json = args.filted_json
    #prediction_json = args.prediction_json
    file_path = args.best_match_json
    ref_file = args.shopsign_json
    match_json = args.match_json
    # 计算平均距离
    avg_distance, count = calculate_average_distance(file_path)
    
    # 打印结果
    print(f"共找到 {count} 个similarity>{sim_threshold}的高德匹配条目")
    print(f"平均距离值为: {avg_distance:.4f} 米")
    
    metrics = evaluate_poi_matches(file_path, ref_file, sim_threshold)
    
    
    # 添加新的匹配准确率评估
    if match_json:
        match_accuracy, correct_count, total_count = calculate_match_accuracy(match_json,prediction_json)
        print(f"\n匹配准确率评估:")
        print(f"准确率: {match_accuracy:.4f} ({correct_count}/{total_count})")
    metrics['total']['precision'] = match_accuracy
    #重新计算F1
    metrics['total']['f1'] = 2 * metrics['total']['precision'] * metrics['total']['recall'] / (metrics['total']['precision'] + metrics['total']['recall']) if (metrics['total']['precision'] + metrics['total']['recall']) > 0 else 0

    print("\n总体评估指标:")
    print(f"精确率: {metrics['total']['precision']:.4f}, 召回率: {metrics['total']['recall']:.4f}, F1: {metrics['total']['f1']:.4f}")
    print(f"TP: {metrics['total']['tp']}, FP: {metrics['total']['fp']}, FN: {metrics['total']['fn']}")

    '''
    print("\nmanual匹配评估指标:")
    print(f"精确率: {metrics['manual']['precision']:.4f}, 召回率: {metrics['manual']['recall']:.4f}, F1: {metrics['manual']['f1']:.4f}")
    print(f"TP: {metrics['manual']['tp']}, FP: {metrics['manual']['fp']}, FN: {metrics['manual']['fn']}")
    
    print("\nGD匹配评估指标:")
    print(f"精确率: {metrics['gd']['precision']:.4f}, 召回率: {metrics['gd']['recall']:.4f}, F1: {metrics['gd']['f1']:.4f}")
    print(f"TP: {metrics['gd']['tp']}, FP: {metrics['gd']['fp']}, FN: {metrics['gd']['fn']}")
    '''
    print("评估完毕")