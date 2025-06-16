# coding=utf-8
# 计算相机的旋转和平移向量
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj
from pyproj import Proj, transform
import warnings
warnings.filterwarnings('ignore')
def ecef2gps(point3d):
    tx, ty, tz = point3d
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 执行转换
    lng, lat, alt = transform(ecef, wgs84, tx, ty, tz,
                              radians=False)  # 是否用弧度返回值
    return [lng, lat, alt]
def gps2ecef(point3d):
    lat,lng, alt = point3d
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 执行转换
    x, y ,z= transform(wgs84,ecef,lng,lat,alt)  # 是否用弧度返回值
    return np.array([x,y,z])

def camera_rotation(rotation_angle):
    rotation = R.from_euler('y', rotation_angle, degrees=True)
    return rotation.as_quat(),rotation.as_matrix()

#output：t[TX, TY, TZ],q[QW, QX, QY, QZ]
def camera_pose(camera_gps):
    camera_ecef = gps2ecef(camera_gps[0:3])
    rotation_quat,rotation_matrix = camera_rotation(camera_gps[3])
    rotation_quat = [rotation_quat[3],rotation_quat[0],rotation_quat[1],rotation_quat[2]]
    t = -np.dot(rotation_matrix, camera_ecef)
    return t,rotation_quat

def enu2ecef(camera_pose1):
    qw, qx, qy, qz = camera_pose1[0:4]
    tx,ty,tz = camera_pose1[4:7]
    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()
    camera_center = -np.dot(rotation_matrix.T, np.array([tx, ty, tz]))  # ecef坐标系
    return camera_center



if  __name__ == "__main__":
    camera_gps = [22.29813248,114.1720733,9.419773102,365.2584229]
    camera_input = gps2ecef(camera_gps[0:3])
    T,Q = camera_pose(camera_gps)
    camera_pose1 = [0.16185830573598994,0.97556944384470934,0.12503295113039262,-0.080205425205682288,1136011.8386793742,6270508.0323905572,178825.11087941995]
    camera_output = enu2ecef(camera_pose1)
    distance = np.linalg.norm(camera_input - camera_output)
    print(distance)



    # 0.99944507478722677 - 0.0014474523302495004 - 0.031457441396469665 - 0.010857105788105878
    # - 0.0070689546936569311 0.054926769338929649 - 6.2393371070153023




