import subprocess
#colmap稀疏重建命令
import os
import shutil
import config.config_signboard_entity as config


def mapper(database_path, image_path, output_path):
    command = [
        r"..\COLMAP-3.8-windows-cuda\COLMAP.bat",
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_path,
        "--output_path",
        output_path,
        "--Mapper.multiple_models",
        "True",
        "Mapper.max_num_models",
        "1"
    ]
    subprocess.run(command,stdout=subprocess.PIPE)

def aligner(database_path,image_path,input_path,output_path,ref_images_path,colmap_path,group_id):
    # --ref_is_gps 1 --alignment_type ecef --robust_alignment 1 --robust_alignment_max_error 3.0
    os.makedirs(output_path, exist_ok=True)
    command1 = [
        r"..\COLMAP-3.8-windows-cuda\COLMAP.bat",
        "model_aligner",
        "--input_path",
        input_path,
        "--output_path",
        output_path,
        "--ref_images_path",
        ref_images_path,
        "--ref_is_gps",
        "1",
        "--alignment_type",
        "ecef",
        "--robust_alignment",
        "1",
        "--robust_alignment_max_error",
        "3.0"
    ]
    command2 = [
        r"..\COLMAP-3.8-windows-cuda\COLMAP.bat",
        "model_converter",
        "--input_path",
        output_path,
        "--output_path",
        output_path,
        "--output_type",
        "TXT"
    ]
    subprocess.run(command1,stdout=subprocess.PIPE)
    if os.path.exists(os.path.join(output_path,"cameras.bin")):
        subprocess.run(command2, stdout=subprocess.PIPE)
    else:
        print("group_id {} reconstruction failed".format(group_id))

def check_points3D_bin_folder(colmap_path):
    max_size = 0
    max_folder = None
    for folder_name in os.listdir(colmap_path):
        folder_path = os.path.join(colmap_path, folder_name)
        if os.path.isdir(folder_path) and folder_name.isdigit():
            points3D_bin_path = os.path.join(folder_path, 'points3D.bin')
            if os.path.exists(points3D_bin_path):
                file_size = os.path.getsize(points3D_bin_path)
                if file_size > max_size:
                    max_size = file_size
                    max_folder = folder_name
    return max_folder

def remapper(database_path,image_path, output_path):
    print("multiple_models")
    command = [
        r"..\COLMAP-3.8-windows-cuda\COLMAP.bat",
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_path,
        "--output_path",
        output_path,
        "--Mapper.multiple_models",
        "True",
        "--Mapper.max_num_models",
        "3"
    ]
    print(command)
    subprocess.run(command, stdout=subprocess.PIPE)


if __name__ == '__main__':
    args =config.parse_args()
    group_path = args.group_path
    image_path = args.img_path
    ref_txt = args.ref_txt
    groups = os.listdir(group_path)
    groups = sorted(list(map(int, groups)))
    for group_id in groups:
        group_id = str(group_id)
        colmap_path = os.path.join(os.path.join(group_path,group_id),'colmap')
        ref_images_path = os.path.join(colmap_path,ref_txt)
        database_path = os.path.join(colmap_path, 'database.db')
        align_path = os.path.join(colmap_path, 'aligned') #存储稀疏重建aligned输出
        
        mapper(database_path,image_path,colmap_path)
        if not os.path.exists(os.path.join(colmap_path, '0')):
            continue
        align_basepath = check_points3D_bin_folder(colmap_path)

        reconstruct_path = os.path.join(colmap_path, align_basepath)  
        if os.path.exists(reconstruct_path):
            aligner(database_path,image_path,reconstruct_path, align_path, ref_images_path,colmap_path,group_id) #坐标系对齐




