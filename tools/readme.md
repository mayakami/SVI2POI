各个文件说明

1.`Equirec2Perspec-master/Equirec2Perspec.py`：投影变换，球面坐标转平面坐标，每张图片切分8张fov=60的水平视角图片（有5°的仰角）。

2.`mycode/read_metadata.py`,读取metadata文件夹,输出是image_gps.csv和metadata_gps.csv,一个是全景图片的位置记录,用来在arcgis中可视化街景拍摄点,一个是给colmap提供位置参考。

note：所有写入csv的程序，encoding='utf-8-sig'，便于导入arcgis

3.`mycode/geocoding.py`：高德的坐标系（gcj02）与goole坐标系（wgs84）对齐的函数

4.`mycode/poi_compare.py`：



