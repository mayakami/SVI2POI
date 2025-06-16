import subprocess
import os
import config.config_signboard_entity as config

#至少需要24G内存才能运行
def run_evaluation(args):
    """
    运行ChatGLM-6B predict poi name
    
    参数:
        pre_seq_len (int): 预训练序列长度
        checkpoint (str): 检查点名称
        step (int): 步骤数
        cuda_device (str): 使用的CUDA设备
        validation_file (str): 验证文件路径
        test_file (str): 测试文件路径
        prompt_column (str): 提示列名
        response_column (str): 响应列名
        model_path (str): 模型路径
        max_source_length (int): 最大源长度
        max_target_length (int): 最大目标长度
        batch_size (int): 评估批次大小
        quantization_bit (int): 量化位数
    
    返回:
        subprocess.CompletedProcess: 命令执行的结果
    """
    pre_seq_len=256
    checkpoint="adgen-chatglm-6b-pt-256-2e-2"
    subject_path = os.path.join(args.streetview_path, "tools", "ChatGLM-6B")
    step=3000
    cuda_device="0"
    validation_file=args.dev_json
    test_file=args.dev_json
    prompt_column="content"
    response_column="summary"
    model_path=f"{subject_path}/model"
    max_source_length=256
    max_target_length=64
    batch_size=1
    quantization_bit=4

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    
    # 构建输出目录
    output_dir = f"/home/moss/streetview_segment/poi_dataset"
    ptuning_checkpoint = f"{subject_path}/ptuning/output/adgen-chatglm-6b-pt-256-2e-2"
    
    # 构建命令
    cmd = [
        "python3", f"{subject_path}/ptuning/main.py",
        "--do_predict",
        f"--validation_file={validation_file}",
        f"--test_file={test_file}",
        "--overwrite_cache",
        f"--prompt_column={prompt_column}",
        f"--response_column={response_column}",
        f"--model_name_or_path={model_path}",
        f"--ptuning_checkpoint={ptuning_checkpoint}",
        f"--output_dir={output_dir}",
        "--overwrite_output_dir",
        f"--max_source_length={max_source_length}",
        f"--max_target_length={max_target_length}",
        f"--per_device_eval_batch_size={batch_size}",
        "--predict_with_generate",
        f"--pre_seq_len={pre_seq_len}",
        f"--quantization_bit={quantization_bit}"
    ]
    
    # 执行命令
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=f"{subject_path}/ptuning/",
            check=True,
            text=True,
            capture_output=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print("子进程执行失败，错误码:", e.returncode)
        print("标准输出:", e.stdout)
        print("标准错误:", e.stderr)
        raise
    

# 示例调用
if __name__ == "__main__":
    args = config.parse_args()
    result = run_evaluation(args)
    print(result.stdout)