# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

import sys
import csv
import sqlite3
import numpy as np
import warnings
import os
import argparse
from camera_extrinsic import camera_pose
from pyproj import Proj, transform

warnings.filterwarnings('ignore')


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2 ** 31 - 1
ref_gps = 'image_gps.txt'  # 存储metadata中的gps坐标，格式：0_back.jpg 22.29790260785873 114.172088201884 9.199027061462402

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase) #生成自定义类型的实例

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(
            CREATE_CAMERAS_TABLE
        )
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        prior_q=np.full(4, np.NaN),
        prior_t=np.full(3, np.NaN),
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )

def access_existing_database(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default=path)
    args = parser.parse_args()

    if not os.path.exists(args.database_path):
        print("ERROR: Database path does not exist.")
        return

    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)



    cameras = db.execute("SELECT * FROM cameras").fetchall() #查询数据
    # Read and check cameras.
    for camera in cameras:
        camera_id, model, width, height, params, prior_focal_length = camera #params是内参,prior_focal_length=0表示不信任预先给定的焦距
        params = blob_to_array(params, np.float64)

    # Read and check keypoints
    #注意:keypoints表格中每三行的第一行才是特征点的像素坐标值，二三两行具体作用未知
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints WHERE rows>=2") #防止出现nonetype的情况
    )

    # Read and check matches(这里是每一帧与其他帧匹配结果,特征点索引对)
    # a = db.execute("SELECT pair_id, data FROM matches").fetchall()
    matches = {
        pair_id_to_image_ids(pair_id): blob_to_array(data, np.uint32, (-1, 2)) #data列数为2,分别存储匹配点在帧1和2的索引
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches WHERE rows>=2")
    }

    # Read and check images
    images ={
        name: image_id
        for name, image_id in db.execute("SELECT name, image_id FROM images")
    }

    two_view_geometries = {
        pair_id_to_image_ids(pair_id):{'data':blob_to_array(data,np.float64),'config':config,'F':blob_to_array(F,np.float64,(3,3)),'E':blob_to_array(F,np.float64,(3,3)),'H':blob_to_array(F,np.float64,(3,3))}
        for pair_id, data, config,F,E,H in db.execute("SELECT pair_id, data,config,F,E,H FROM two_view_geometries WHERE rows>=2")
    }
    # Read and check images
    print("#")



def example_usage():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database0.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    model2, width2, height2, params2 = (
        2,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0, 0.1)),
    )

    camera_id1 = db.add_camera(model1, width1, height1, params1) #会自己分配camera_id
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1) ##会自己分配image_id
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.

    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)
    db.add_two_view_geometry(image_id1, image_id2, matches12)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2) #params和params2基本相同

    # Read and check keypoints.

    two_view_geometries = {
        pair_id_to_image_ids(pair_id):{'data':blob_to_array(data,np.uint32),'config':config,'F':blob_to_array(F,np.float64,(3,3)),'E':blob_to_array(F,np.float64,(3,3)),'H':blob_to_array(F,np.float64,(3,3))}
        for pair_id, data, config,F,E,H in db.execute("SELECT pair_id, data,config,F,E,H FROM two_view_geometries WHERE rows>=2")
    }

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [
        image_ids_to_pair_id(*pair)
        for pair in (
            (image_id1, image_id2),
            (image_id2, image_id3),
            (image_id3, image_id4),
        )
    ]

    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)

def gps2ecef(point3d):
    lat, lng, alt = point3d
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # 执行转换
    x, y ,z= transform(wgs84,ecef,lng,lat,alt)
    return [x,y,z]

#创建新的database.db
def create_database(new_data,cameras,dir_path):
    colmap_dir = os.path.join(dir_path, 'colmap')
    if os.path.exists(new_data):
        os.remove(new_data)
    # Open the database.
    db1 = COLMAPDatabase.connect(new_data)
    db1.create_tables()

    #r and w camera
    for camera in cameras:
        camera_id, model, width, height, params = camera #params是内参
        camera_id = db1.add_camera(model, width, height, params) #默认只有一个camera_id

    file_ls = []
    name_id = {}#存储image_id和image_name的对应关系
    csv_path = os.path.join(dir_path, 'image_gps.csv')
    data_dict = {}
    ref_gps = os.path.join(colmap_dir, 'image_gps.txt')
    if os.path.exists(ref_gps):
        os.remove(ref_gps)
    with open(csv_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 列标题
        for row in reader:
            with open(ref_gps, 'a') as file:
                file.write( ' '.join(map(str, row)))
                file.write('\n')
            camera_gps = [float(item) for item in row[1:]]
            prior_t,prior_q = camera_pose(camera_gps)
            data_dict[row[0]] = camera_gps
            file_ls.append(row[0])
            image_id = db1.add_image(row[0], camera_id, prior_q, prior_t)
            name_id[row[0]] = image_id

    #读取'image_gps.txt',转ecef坐标系,并存入coordinates中，格式为{image_id:[-2417585.2855302   5386387.60670353  2404969.54242236]}
    coordinates = {} #image_id是从1开始的,所以只能构造成字典


    with open(os.path.join(dir_path,'keypoints.txt'), 'r') as file:
        for line in file:
            # 去除行末的换行符并以冒号分割行
            line = line.strip()
            parts = line.split(':')
            # image_name_pairs = [f"{image_names[i]}_{image_names[i + 1]}.jpg" for i in range(0, len(image_names), 2)]
            image_id = name_id[parts[0]]
            kp = list(map(float, parts[1].split(' ')))
            kp = np.array([[kp[i], kp[i + 1]] for i in range(0, len(kp), 2)])
            db1.add_keypoints(image_id, kp)

    count_match = 0
    with open(os.path.join(dir_path,'matches.txt'), 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            match_id = parts[0].split(' ')
            # print(match_id)
            match_id = [name_id[i] for i in match_id] #name_id
            if parts[1] == '': #没有匹配结果
                continue
                # match_idx = np.array([])
            else:
                match_idx = list(map(float, parts[1].split(' ')))
                match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
                # distance =  np.linalg.norm(coordinates[match_id[0]]-coordinates[match_id[1]])
                # if (len(match_idx)<200) or (distance>25):
                if (len(match_idx) < 200):
                    count_match+=1
                    continue
                db1.add_matches(match_id[0], match_id[1], match_idx)
                db1.add_two_view_geometry(match_id[0], match_id[1], match_idx)
    # print("跳过的匹配图像对:" + str(count_match))
    db1.commit()

    # Read and check keypoints.
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db1.execute("SELECT image_id, data FROM keypoints")
    )
    # Read and check matches.
    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db1.execute("SELECT pair_id, data FROM matches")
    )
    # Read and check matches.
    two_view_geometries = {
        pair_id_to_image_ids(pair_id):{'data':blob_to_array(data,np.uint32),'config':config,'F':blob_to_array(F,np.float64,(3,3)),'E':blob_to_array(F,np.float64,(3,3)),'H':blob_to_array(F,np.float64,(3,3))}
        for pair_id, data, config,F,E,H in db1.execute("SELECT pair_id, data,config,F,E,H FROM two_view_geometries WHERE rows>=2")
    }
    # Create dummy images.
    db1.close()

def get_table_schema(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default=path)
    args = parser.parse_args()

    if not os.path.exists(args.database_path):
        print("ERROR: Database path does not exist.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)
    # 查询db中所有键名
    cursor = db.cursor() # 创建游标对象
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall() #[('cameras',), ('sqlite_sequence',), ('images',), ('keypoints',), ('descriptors',), ('matches',), ('two_view_geometries',)]

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema_info = cursor.fetchall()
        print(f"Schema for table '{table_name}':")
        for column in schema_info:
            print(f"  Column name: {column[1]}, Data type: {column[2]}")
    cursor.close()

def create_database(new_data,cameras,dir_path):
    colmap_dir = os.path.join(dir_path, 'colmap')
    if os.path.exists(new_data):
        os.remove(new_data)
    # Open the database.
    db1 = COLMAPDatabase.connect(new_data)
    db1.create_tables()

    #r and w camera
    for camera in cameras:
        camera_id, model, width, height, params = camera #params是内参
        camera_id = db1.add_camera(model, width, height, params) #默认只有一个camera_id

    file_ls = []
    name_id = {}#存储image_id和image_name的对应关系
    csv_path = os.path.join(dir_path, 'image_gps.csv')
    data_dict = {}
    ref_gps = os.path.join(colmap_dir, 'image_gps.txt')
    if os.path.exists(ref_gps):
        os.remove(ref_gps)
    with open(csv_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 列标题
        for row in reader:
            with open(ref_gps, 'a') as file:
                file.write( ' '.join(map(str, row)))
                file.write('\n')
            camera_gps = [float(item) for item in row[1:]]
            prior_t,prior_q = camera_pose(camera_gps)
            data_dict[row[0]] = camera_gps
            file_ls.append(row[0])
            image_id = db1.add_image(row[0], camera_id, prior_q, prior_t)
            name_id[row[0]] = image_id

    #读取'image_gps.txt',转ecef坐标系,并存入coordinates中，格式为{image_id:[-2417585.2855302   5386387.60670353  2404969.54242236]}
    coordinates = {} #image_id是从1开始的,所以只能构造成字典


    with open(os.path.join(dir_path,'keypoints.txt'), 'r') as file:
        for line in file:
            # 去除行末的换行符并以冒号分割行
            line = line.strip()
            parts = line.split(':')
            # image_name_pairs = [f"{image_names[i]}_{image_names[i + 1]}.jpg" for i in range(0, len(image_names), 2)]
            image_id = name_id[parts[0]]
            kp = list(map(float, parts[1].split(' ')))
            kp = np.array([[kp[i], kp[i + 1]] for i in range(0, len(kp), 2)])
            db1.add_keypoints(image_id, kp)

    count_match = 0
    with open(os.path.join(dir_path,'matches.txt'), 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            match_id = parts[0].split(' ')
            # print(match_id)
            match_id = [name_id[i] for i in match_id] #name_id
            if parts[1] == '': #没有匹配结果
                continue
                # match_idx = np.array([])
            else:
                match_idx = list(map(float, parts[1].split(' ')))
                match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
                # distance =  np.linalg.norm(coordinates[match_id[0]]-coordinates[match_id[1]])
                # if (len(match_idx)<200) or (distance>25):
                if (len(match_idx) < 200):
                    count_match+=1
                    continue
                db1.add_matches(match_id[0], match_id[1], match_idx)
                db1.add_two_view_geometry(match_id[0], match_id[1], match_idx)
    # print("跳过的匹配图像对:" + str(count_match))
    db1.commit()

    # Read and check keypoints.
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db1.execute("SELECT image_id, data FROM keypoints")
    )
    # Read and check matches.
    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db1.execute("SELECT pair_id, data FROM matches")
    )
    # Read and check matches.
    two_view_geometries = {
        pair_id_to_image_ids(pair_id):{'data':blob_to_array(data,np.uint32),'config':config,'F':blob_to_array(F,np.float64,(3,3)),'E':blob_to_array(F,np.float64,(3,3)),'H':blob_to_array(F,np.float64,(3,3))}
        for pair_id, data, config,F,E,H in db1.execute("SELECT pair_id, data,config,F,E,H FROM two_view_geometries WHERE rows>=2")
    }
    # Create dummy images.
    db1.close()

def create_database_test(new_data,cameras,dir_path):
    save_images = []
    colmap_dir = os.path.join(dir_path, 'colmap')
    input_dir = os.path.join(colmap_dir,'input')
    os.makedirs(input_dir,exist_ok='True')
    if os.path.exists(new_data):
        os.remove(new_data)
    # Open the database.
    db = COLMAPDatabase.connect(new_data)
    db.create_tables()

    #r and w camera CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]
    with open(os.path.join(input_dir,'cameras.txt'),'w')  as file:
        for camera in cameras:
            camera_id, model, width, height, params = camera #params是内参
            model_name = 'PINHOLE' if model == 1 else 'UNKNOWN'
            parameters_str = ' '.join(map(str, params))
            file.write(f"{camera_id} {model_name} {width} {height} {parameters_str}\n")
            camera_id = db.add_camera(model, width, height, params) #默认只有一个camera_id

    file_ls = []
    data_dict = {}
    name_id = {}#{image_name:image_id}
    csv_path = os.path.join(dir_path, 'image_gps.csv')
    ref_gps = os.path.join(colmap_dir, 'image_gps.txt')
    if os.path.exists(ref_gps):
        os.remove(ref_gps)
    if os.path.exists(os.path.join(input_dir,'images.txt')):
        os.remove(os.path.join(input_dir,'images.txt'))
    with open(csv_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 列标题
        for row in reader:
            with open(ref_gps, 'a') as file:
                file.write( ' '.join(map(str, row[:-1])))
                file.write('\n')
            camera_gps = [float(item) for item in row[1:]]
            prior_t,prior_q = camera_pose(camera_gps)
            data_dict[row[0]] = camera_gps
            file_ls.append(row[0])
            image_id = db.add_image(row[0], camera_id, prior_q, prior_t)
            name_id[row[0]] = image_id
            Q_str = ' '.join(map(str, prior_q))
            T_str = ' '.join(map(str, prior_t))
            save_images.append(f"{image_id} {Q_str} {T_str} {camera_id} {row[0]}\n")
            with open(os.path.join(input_dir,'images.txt'),'a')  as file: #IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                Q_str = ' '.join(map(str, prior_q))
                T_str = ' '.join(map(str, prior_t))
                file.write(f"{image_id} {Q_str} {T_str} {camera_id} {row[0]}\n")
                file.write(f"\n")
    #读取'image_gps.txt',转ecef坐标系,并存入coordinates中，格式为{image_id:[-2417585.2855302   5386387.60670353  2404969.54242236]}
    coordinates = {} #image_id是从1开始的,所以只能构造成字典
    with open(os.path.join(dir_path,'keypoints.txt'), 'r') as file:
        for line in file:
            # 去除行末的换行符并以冒号分割行
            line = line.strip()
            parts = line.split(':')
            # image_name_pairs = [f"{image_names[i]}_{image_names[i + 1]}.jpg" for i in range(0, len(image_names), 2)]
            image_id = name_id[parts[0]]
            kp = list(map(float, parts[1].split(' ')))
            kp = np.array([[kp[i], kp[i + 1]] for i in range(0, len(kp), 2)])
            db.add_keypoints(image_id, kp)

    count_match = 0
    with open(os.path.join(dir_path,'matches.txt'), 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(':')
            match_id = parts[0].split(' ')
            # print(match_id)
            match_id = [name_id[i] for i in match_id] #name_id
            if parts[1] == '': #没有匹配结果
                continue
                # match_idx = np.array([])
            else:
                match_idx = list(map(float, parts[1].split(' ')))
                match_idx = np.array([[match_idx[i], match_idx[i + 1]] for i in range(0, len(match_idx), 2)])
                # distance =  np.linalg.norm(coordinates[match_id[0]]-coordinates[match_id[1]])
                # if (len(match_idx)<200) or (distance>25):
                if (len(match_idx) < 200):
                    count_match+=1
                    continue
                db.add_matches(match_id[0], match_id[1], match_idx)
                db.add_two_view_geometry(match_id[0], match_id[1], match_idx)
    # print("跳过的匹配图像对:" + str(count_match))
    db.commit()

    # Read and check keypoints.
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute("SELECT image_id, data FROM keypoints")
    )
    # Read and check matches.
    matches = dict(
        (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )
    # Read and check matches.
    two_view_geometries = {
        pair_id_to_image_ids(pair_id):{'data':blob_to_array(data,np.uint32),'config':config,'F':blob_to_array(F,np.float64,(3,3)),'E':blob_to_array(F,np.float64,(3,3)),'H':blob_to_array(F,np.float64,(3,3))}
        for pair_id, data, config,F,E,H in db.execute("SELECT pair_id, data,config,F,E,H FROM two_view_geometries WHERE rows>=2")
    }
    # Create dummy images.
    db.close()










if __name__ == "__main__":

    existing_data = 'database.db'
    new_data = 'database1.db'
    # get_table_schema(existing_data) #来获取database.db中各个子表的结构信息
    example_usage()
    # access_existing_database(existing_data) #读取已有的database.db
