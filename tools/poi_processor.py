import json
import os
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(__file__)
import config.config_signboard_entity as config
import re  # 添加正则表达式模块

def merge_poi_files(group_path, group_dirs, poi_filename):# -> dict:
    """
    合并多个组目录下的特定POI文件
    """
    all_pois = {}
    poi_count = 0
    
    print(f"合并POI文件的基础路径: {group_path}")
    print(f"要处理的组目录数量: {len(group_dirs)}")
    print(f"POI文件名: {poi_filename}")
    
    for group_id in group_dirs:
        # 获取特定的POI文件路径
        group_dir = os.path.join(group_path, group_id)
        poi_file = os.path.join(group_dir, poi_filename)
        if os.path.exists(poi_file):
            try:
                with open(poi_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for _, poi in data.items():
                    # 重新编号POI
                    all_pois[str(poi_count)] = poi
                    poi_count += 1  
                print(f"已加载POI文件: {poi_file}")
            except Exception as e:
                print(f"处理文件 {poi_file} 时出错: {e}")
    print(f"总共合并了 {poi_count} 个POI项目")
    return all_pois

def load_reference_labels(ref_file):
    """
    加载参考标签文件
    返回一个字典：{sign_id: poi_info}，用于查找每个sign_id对应的POI信息
    """
    # 检查文件是否存在
    if not os.path.exists(ref_file):
        print(f"警告: 参考标签文件 {ref_file} 不存在，将使用默认标签")
        return {}
    
    with open(ref_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建sign_id到POI的映射
    sign_to_poi = {}
    
    # 遍历所有POI
    for poi in data.get("poi", []):
        poi_name = poi.get("name", "")
        poi_id = poi.get("poi_id")
        sign_group = poi.get("sign_group", [])
        
        # 为每个sign_id建立到POI的映射
        for sign_id in sign_group:
            sign_to_poi[sign_id] = {
                "name": poi_name,
                "poi_id": poi_id
            }
    
    #print(f"已加载 {len(sign_to_poi)} 个sign_id映射关系")
    return sign_to_poi

def match_labels(pois, signs_by_id):
    """
    匹配POI与真实标签
    """
    labeled_pois = {}
    # 检查是否有参考标签
    has_references = len(signs_by_id) > 0
    
    for poi_id, poi in pois.items():
        if has_references:
            # 获取该POI的sign_id列表
            sign_ids = poi.get("sign_id", [])
        
            # 统计最常见的name和对应的原始poi_id
            name_counts = defaultdict(int)
            name_to_original_poi_id = {}  # 存储name对应的原始poi_id
        
            for sign_id in sign_ids:
                if sign_id in signs_by_id:
                    name = signs_by_id[sign_id].get("name", "")
                    original_poi_id = signs_by_id[sign_id].get("poi_id")
                    
                    if name:
                        name_counts[name] += 1
                        # 保存每个name对应的原始poi_id
                        if name not in name_to_original_poi_id:
                            name_to_original_poi_id[name] = original_poi_id
        
            # 找出出现次数最多的name作为真实值
            best_match = max(name_counts.items(), key=lambda x: x[1], default=(None, 0))
            best_name = best_match[0] if best_match[0] else "未知"
            # 获取best_name对应的原始poi_id，如果没有则使用默认值
            original_poi_id = name_to_original_poi_id.get(best_name, -1)
            best_name = re.sub(r'\([^)]*\)', '', best_name).strip() # 去除括号内的内容
        else:
            # 没有参考标签，使用默认值
            best_name = "未知"
            original_poi_id = -1

        # 复制POI信息并添加label字段
        new_poi = poi.copy()
        new_poi["label"] = {
            "poi_id": original_poi_id,
            "name": best_name
        }
        labeled_pois[poi_id] = new_poi
    
    return labeled_pois

def generate_dataset(labeled_pois):
    """
    生成指定格式的数据集
    """
    dataset = []
    
    for poi_id, poi in labeled_pois.items():
        # 获取text_content并用空格连接
        text_content = " ".join(poi.get("text_content", [])).replace('"', '').replace("'", '')
        has_chinese = bool(re.search('[\u4e00-\u9fff]', text_content))

        # 获取真实值
        true_name = poi.get("label", {}).get("name", "未知")
        if has_chinese:
            # 包含中文字符，使用原始prompt
            prompt = prompt = (
                f"你是一名经验丰富的语言学家，你的任务是从下列OCR文本中提取POI实体名称。"
                f"这些文本来自同一地点的多个招牌OCR结果，其中可能包含错误识别或重复信息。"
                f"输出应只包含OCR文本中确实存在的信息组合，不要凭空添加任何额外内容: {text_content}"
            )
        else:
             # 不包含中文字符，添加英文示例
            prompt = (
                f"你是一名经验丰富的语言学家，你的任务是从下列OCR文本中提取POI实体名称。"
                f"这些文本来自同一地点的多个招牌OCR结果，其中可能包含错误识别或重复信息。"
                f"输出应只包含OCR文本中确实存在的信息组合，不要凭空添加任何额外内容: {text_content}"
                f" 示例输入1：EASTLAND Since 1955,示例输出1：EAST LAND，"
                f" 示例输入2：PARE KOTA ADIPURA BANK JATIM,示例输出2：BANK JATIM"
            )
     
        # 创建数据集条目
        entry = {
            "point_seq_id": poi_id,
            "content": prompt,
            "summary": true_name
        }
        
        dataset.append(entry)
    
    print(f"已生成 {len(dataset)} 条数据集记录")
    return dataset

def save_json(data, output_file):
    """
    保存JSON数据到文件
    """
    print(f"正在保存数据到文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已成功保存到: {output_file}")

def process_poi_data(args):
    """
    合并多个目录下的POI文件，加载参考标签（可以没有），以及生成用于POI名称提取的数据集
    
    参数:
        args: 配置参数对象，包含以下字段:
            - group_path: 组数据根目录，包含多个组子目录
            - poi_json: POI文件名(在每个组目录下)
            - shopsign_json: 参考标签文件路径
            - merged_poi_json: 合并后的POI数据输出路径
            - dev_json: 生成的数据集输出路径
    """
    group_path = args.group_path
    groups = os.listdir(group_path)
    poi_path = args.poi_json
    ref_file = args.shopsign_json
    merged_output = args.merged_poi_json
    dataset_output = args.dev_json

    print("步骤1: 合并POI文件...")
    print(f"组路径: {group_path}")
    print(f"POI文件路径: {poi_path}")
    merged_pois = merge_poi_files(group_path, groups, poi_path)
    
    # 步骤2: 加载参考标签
    print("步骤2: 加载参考标签...")
    print(f"参考标签文件: {ref_file}")
    signs_by_id = load_reference_labels(ref_file) #{sign_id:{'name': '太子珠宝钟表(汉口道)', 'poi_id': 0}}

    # 步骤3: 匹配标签
    print("步骤3: 匹配标签...")
    labeled_pois = match_labels(merged_pois, signs_by_id)
    
    # 步骤4: 保存合并的POI数据
    print("步骤4: 保存合并的POI数据...")
    print(f"合并输出路径: {merged_output}")
    save_json(labeled_pois, merged_output)

    # 步骤5: 生成数据集
    print("步骤5: 生成数据集...")
    print(f"数据集输出路径: {dataset_output}")
    dataset = generate_dataset(labeled_pois)

    with open(dataset_output, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"数据集已保存至: {dataset_output}")

if __name__ == "__main__":
    # 设置路径
    args = config.parse_args()
    process_poi_data(args)