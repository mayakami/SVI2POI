"""
高德在国内的坐标系需要和wgs坐标系（GPS）对齐
"""
#from ecef2gps import gps2ecef
import numpy as np
import hashlib
from urllib import parse
import math
import requests
import csv
import os
x_pi = 3.14159265358979324 * 3000.0 / 180.0
# 圆周率π
pi = 3.1415926535897932384626
# 长半轴长度
a = 6378245.0
# 地球的角离心率
ee = 0.00669342162296594323
# 矫正参数
interval = 0.000001
from zhconv import convert

def read_csv_to_dict(filename,type):
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        name_list = []
        poi_list = []
        for row in reader:
            name = row['\ufeff名称']
            name = convert(name, 'zh-hans')
            longitude = float(row['经度'])
            latitude = float(row['纬度'])
            if type == 'gcj02':
                lat_wgs,lon_wgs = gcj02_to_wgs84(longitude,latitude)
                row['经度'] = lon_wgs
                row['纬度'] = lat_wgs
            #将每个POI的字典添加到列表中
            poi_list.append(row)
            #记得坐标系转换
            # location_dict[name] = [latitude,longitude]\
    return poi_list


class BD_Geocoding:
    # 基于百度地理编码的sn验证方式，IP白名单验证方式在个人管理页面添加IP即可
    def __init__(self, ak, sk):
        self.ak = ak
        self.sk = sk

    def baidu_geocode(self, address):
        """
        利用百度geocoding服务解析地址获取位置坐标
        :param address:需要解析的地址
        :return:
        """

        url = 'http://api.map.baidu.com'
        queryStr = '/geocoder/v2/?address={address}&output=json&ak={ak}'.format(address=address, ak=self.ak)
        encodeStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
        rawStr = encodeStr + self.sk
        sn = hashlib.md5(parse.quote_plus(rawStr).encode(encoding='utf-8')).hexdigest()
        try:
            response = requests.get(url + queryStr + '&sn={sn}'.format(sn=sn)).json()
            location = response['result']['location']
            lng = location['lng']
            lat = location['lat']
            return [lng, lat]
        except:
            result = ['0', '0']
            return result


def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)：谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:列表返回
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)：百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:列表返回
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gc_lng = z * math.cos(theta)
    gc_lat = z * math.sin(theta)
    return [gc_lng, gc_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:列表返回
    """
    # 判断是否在国内
    if out_of_china(lng, lat):
        return lng, lat
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    gclng = lng + dlng
    gclat = lat + dlat
    return [gclng, gclat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:列表返回
    """
    # 判断是否在国内
    if out_of_china(lng, lat):
        return lng, lat
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    wgslng = lng + dlng
    wgslat = lat + dlat

    #新加误差矫正部分
    corrent_list = wgs84_to_gcj02(wgslng, wgslat)
    clng = corrent_list[0]-lng
    clat = corrent_list[1]-lat
    dis = math.sqrt(clng*clng + clat*clat)

    while dis > interval:
        clng = clng/2
        clat = clat/2
        wgslng = wgslng - clng
        wgslat = wgslat - clat
        corrent_list = wgs84_to_gcj02(wgslng, wgslat)
        cclng = corrent_list[0] - lng
        cclat = corrent_list[1] - lat
        dis = math.sqrt(cclng*cclng + cclat*cclat)
        clng = clng if math.fabs(clng) > math.fabs(cclng) else cclng
        clat = clat if math.fabs(clat) > math.fabs(cclat) else cclat

    return [wgslng,wgslat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)

def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


if __name__ == '__main__':
    #将POI选择范围转为gcj02坐标系
    # wgs_bbox = [114.1708,114.1727,22.2975,22.3087]
    # wgs_bbox = [114.17042491624558, 114.17330024430954,22.29413099947349, 22.30989393266365]
    wgs_bbox = [114.16988805162937,114.17299941408665,22.305850658504372, 22.308103844058174]
    result_max = wgs84_to_gcj02(wgs_bbox[1],wgs_bbox[3]) #右上角
    result_min = wgs84_to_gcj02(wgs_bbox[0],wgs_bbox[2]) #左下角
    gcj_bbox = [result_min[0],result_max[0],result_min[1],result_max[1]]
    print(gcj_bbox)
    # result1 = gcj02_to_wgs84(114.176239,22.302913)


    gd_poi_gcj =[114.176141,22.298598]
    gd_poi_wgs = gcj02_to_wgs84(gd_poi_gcj[0],gd_poi_gcj[1])
    print("gd_poi_wgs: " + str(gd_poi_wgs))
    #114.176522, 22.301106
    extracted_poi_wgs = [114.1715031,22.30392998]
    extracted_poi_gcj = wgs84_to_gcj02(extracted_poi_wgs[0],extracted_poi_wgs[1])
    print("extracted_poi_wgs: " + str(extracted_poi_wgs))


    # #从高德爬取的poi需要做坐标系转换
    # filename = "../../POI_ref/poi_gd.csv"
    # poi_wgs = read_csv_to_dict(filename,'gcj02')
    # new_filename = "../../POI_ref/poi_gd_wgs.csv"
    # with open(new_filename, mode='w', newline='', encoding='utf-8-sig') as csvfile:
    #     if poi_wgs:
    #         fieldnames = poi_wgs[0].keys()  # 获取列名
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         if os.stat(new_filename).st_size == 0:
    #             writer.writeheader()  # 创建表头
    #         for row in poi_wgs:
    #             writer.writerow(row)
    # print('done')
    #
    #





#     result1 = gcj02_to_bd09(lng, lat)
#     result2 = bd09_to_gcj02(lng, lat)
#     result3 = wgs84_to_gcj02(lng, lat)
#     result4 = gcj02_to_wgs84(lng, lat)
#     result5 = bd09_to_wgs84(lng, lat)
#     result6 = wgs84_to_bd09(lng, lat)
#
#     # 填写在百度生成应用时的ak和sk
#     bd_geo = BD_Geocoding('ak', 'sk')
#     result7 = bd_geo.baidu_geocode('北京市朝阳区宏源大厦')
#     print(result1, result2, result3, result4, result5, result6, result7)
