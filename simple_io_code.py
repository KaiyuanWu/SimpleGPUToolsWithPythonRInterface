# -*- coding: utf-8 -*-
# 将图片数据存储到leveldb中
import cv2
import numpy as np
import leveldb
import logging
import struct
FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT)
#filename是一个列表文件，里面包含需要存储的图片列表     
#idx id image_filename additional_label
def cvt_data_to_leveldb(filename,  leveldb_dir, addtional_label_length=0):
    db = leveldb.LevelDB(leveldb_dir,write_buffer_size = 536870912)
    count = 0
    for line in open(filename):
        fields = line.split()
        idx = fields[0]
        id = int(fields[1])
        image_filename = fields[2]
        if addtional_label_length > 0:
            addtional_label = np.array([float(i) for i in fields[3:]])
        try:
            img = cv2.imread(image_filename)
            h,w,c = img.shape
            img_str = img.tostring()
            if addtional_label_length > 0:
                addtional_label_str = addtional_label.tostring()
                fmt="i%dsiii%ds%ds"%(len(img_str),len(addtional_label_str),len(image_filename))
                value = struct.pack(fmt, id, img_str,h,w,c, addtional_label_str, image_filename)
            else:
                fmt="i%dsiii%ds"%(len(img_str),len(image_filename))
                value = struct.pack(fmt, id, img_str,h,w,c,  image_filename)
            key = "%s_%s"%(idx, fmt)
            print(key)
            db.Put(key, value)
        except Exception as e:
            logging.error("Error while save %s, error: %s"%(image_filename, e.message))
        if count%10000 == 0:
            logging.warning("processed %d items"%count)
        count += 1    
    logging.warning("finish process all items")

def test_read_leveldb(leveldb_dir, with_additional_label = False):
    db = leveldb.LevelDB(leveldb_dir)
    for key in db.RangeIter(include_value = False):
        value = db.Get(key)
        idx, fmt = key.split('_')
        if with_additional_label:
            id, img_str,h,w,c, addtional_label_str, image_filename = struct.unpack(fmt, value)
        else:
            id, img_str,h,w,c,  image_filename = struct.unpack(fmt, value)
        print(id, h,w,c, image_filename)        
        img = np.zeros((h,w,c), dtype='uint8')
        img.data[:] = img_str
        cv2.imshow('img', img)
        cv2.waitKey(0)
