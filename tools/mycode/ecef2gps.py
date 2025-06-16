import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj
from pyproj import Proj, transform,CRS,Transformer

crs_WGS84 = CRS.from_epsg(4326)  # WGS84地理坐标系
crs_WebMercator = CRS.from_epsg(3857)  # Web墨卡托投影坐标系
cell_size = 0.009330691929342804  # 分辨率（米），一个像素表示的大小(24级瓦片)
origin_level = 24  # 原始瓦片级别
EarthRadius = 6378137.0  # 地球半径
tile_size = 256  # 瓦片大小
import warnings
warnings.filterwarnings('ignore')
def ecef2gps(point3d):
    tx, ty, tz = point3d
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 执行转换
    lng, lat, alt = transform(ecef, wgs84, tx, ty, tz,
                              radians=False)  # 是否用弧度返回值
    return [lat, lng, alt]
def gps2ecef(point3d):
    lat,lng, alt = point3d
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 执行转换
    x, y ,z= transform(wgs84,ecef,lng,lat,alt)  # 是否用弧度返回值
    return np.array([x,y,z])


def WGS84ToWebMercator(lat_lon):
    """
    WGS84坐标转web墨卡托坐标
    :param lat_lon:  纬度,经度集合
    :return:  web墨卡托坐标x,y集合
    """
    transformer = Transformer.from_crs(crs_WGS84, crs_WebMercator)
    x_y = list(transformer.itransform(lat_lon))
    return x_y
def WebMercator2WGS84(x_y):
    """
    web墨卡托坐标转WGS84坐标
    :param x_y:  web墨卡托坐标x,y集合
    :return:  纬度,经度集合
    """
    transformer = transform.from_crs(crs_WebMercator, crs_WGS84)
    lat_lon = transformer.itransform(x_y)
    return lat_lon

if __name__ == "__main__":
    # 四元数 (QW, QX, QY, QZ)
    qw, qx, qy, qz = 0.12596032655925199,0.081191086144761615,-0.16321475275150291,0.975142527094366   # 替换为实际值
    #注意，这里的pose(7)是先平移后旋转，所以需要乘以T
    # 平移向量 (TX, TY, TZ)
    tx, ty, tz = -1124959.7362887501,6273045.902538809,-158524.03845527643 #t
    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    camera_center = -np.dot(rotation_matrix.T, np.array([tx, ty, tz])) #X_world
    print("投影中心坐标:（ecef坐标系下）", camera_center)
    rotation = R.from_euler('z', 185.2584229, degrees=True)
    euler = rotation.as_quat()
    #原坐标22.29799725	114.1720826	9.144699097	185.2584229
    lat0, lon0, alt0 =22.29799725,114.1720826,9.144699097

    # 计算相机中心坐标(ECEF坐标系下)
    camera_center = -np.dot(rotation_matrix.T, np.array([tx, ty, tz]))
    print("投影中心坐标:（ecef坐标系下）", camera_center)

    # 定义坐标转换
    # 从ECEF转换到WGS84
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    # 执行转换
    lon, lat, alt = transform(ecef, wgs84, camera_center[0],camera_center[1], camera_center[2], radians=False) #是否用弧度返回值
    # 输出WGS84经纬度坐标
    print("经度:", lon, "纬度:", lat, "高度:", alt)

    x,y,z = transform(wgs84, ecef,lon0,lat0, alt0)
    camera_center0 = np.array([x,y,z])
    distance = np.linalg.norm(camera_center0 - camera_center)
    print("原相机坐标和输出相机坐标距离: ", distance) #output:0.069m


    # 从WGS84转换到UTM Zone 50N
    utm_zone_50n = Proj(proj="utm", zone=50, ellps='WGS84', datum='WGS84', south=False)
    utm_x, utm_y = transform(wgs84, utm_zone_50n, lon, lat)

    # 输出UTM坐标
    print("WGS84坐标系下：UTM Zone 50N X:", utm_x, "Y:", utm_y,"Z:",alt)

    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    # 将旋转矩阵转换为欧拉角 (Omega, Phi, Kappa)
    # 摄影测量中通常使用 ZYX 旋转顺序
    omega, phi, kappa = rotation.as_euler('ZYX', degrees=True)
    print("旋转矩阵:\n", rotation_matrix)

    print("Omega:", omega, "Phi:", phi, "Kappa:", kappa)
