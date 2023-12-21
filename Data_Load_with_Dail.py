import os
import random
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from nvidia import dali
from nvidia.dali import tensors
from nvidia.dali import pipeline_def
from nvidia.dali.fn import experimental
# import nvidia.dali.fn as fn
from nvidia.dali import fn
from PIL import Image
from torchvision import transforms
from PIL import Image
import json
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from tqdm import tqdm
import sys
import datetime
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")
import re

class DataSource(object):
    def __init__(self, Datainf,Data_root_path,SGS_path = 'SGS',DLS_path='DLS_npy',shuffle=True,batch_size=64,load_type='train',seq_number=4):
        self.batch_size = batch_size
        self.paths = list(zip(*(Datainf,)))
        self.shuffle = shuffle
        self.data_root_path = Data_root_path
        self.Satellite_img_file = os.path.join(self.data_root_path, SGS_path)
        self.BSVI_sequence_file = os.path.join(self.data_root_path, DLS_path)
        self.load_type = load_type
        self.seq_number = seq_number

        if shuffle:
            random.shuffle(self.paths)
        pass

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.paths)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.paths):
            self.__iter__()
            raise StopIteration

        """
        数据存放定义
        """
        Satellite_img_list = []
        BSVI_sequence_list = []
        Label_list = []
        if self.load_type == 'cam':
            index_list = []

        for _ in range(self.batch_size):
            data_inf = self.paths[self.i % len(self.paths)][0]
            satellite_img_path = os.path.join(self.Satellite_img_file, data_inf['SGS'])
            Sence_sequence_path_list = [os.path.join(self.BSVI_sequence_file, img_name.split('.')[0] + '.npy') for img_name in data_inf['DLS']]
            Sence_sequence_path_list = Sence_sequence_path_list[:self.seq_number]
            Label = data_inf['Label']

            sequence_list = []
            Satellite_img_list.append(np.fromfile(satellite_img_path, dtype=np.uint8))
            for sequence_index in Sence_sequence_path_list:
                sequence_list.append(np.load(sequence_index))
            sequence_list = np.stack(sequence_list)
            BSVI_sequence_list.append(sequence_list)
            Label_list.append(np.array(Label))
            if self.load_type == 'cam':
                # print('函数阶段打印index为{}'.format(data_inf['satellite']))
                index = re.findall(r'\d+', data_inf['SGS'])[0]
                index_list.append(int(index))
            self.i += 1
        if self.load_type == 'train':
            return (Satellite_img_list,BSVI_sequence_list,Label_list)
        elif self.load_type == 'cam':
            index_list = np.array(index_list)
            return (Satellite_img_list, BSVI_sequence_list, Label_list,index_list)


    def __len__(self):
        return len(self.paths)

    next = __next__

class SourcePipeline(Pipeline):
    def __init__(self,  batch_size, num_threads, device_id, external_data,modeltype,load_type='train'):
        super(SourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     exec_async=True,
                                                     exec_pipelined=True,
                                                     prefetch_queue_depth = 2,
                                                     )
        self.load_type = load_type

        self.input_Scene_img = ops.ExternalSource()
        self.input_Scene_sequence = ops.ExternalSource()
        self.intput_Label = ops.ExternalSource()
        if self.load_type == 'cam':
            self.input_index_list = ops.ExternalSource()

        self.external_data = external_data
        self.model_type = modeltype
        self.iterator = iter(self.external_data)
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.decode_for_depth = ops.decoders.Image(device="mixed", output_type=types.ANY_DATA)
        self.cat = ops.Cat(device="gpu",axis=2)
        self.tran = ops.Transpose(device="gpu",perm=[2,0,1])
        self.crop = ops.RandomResizedCrop(device="gpu",size =256,random_area=[0.8, 1.00])
        self.resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)
        self.Sence_img_resize = ops.Resize(device='gpu', resize_x=320, resize_y=320)
        self.no_mirrror_cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop_pos_x = 0,
                                            crop_pos_y = 0,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255, 0,0,0],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255, 1,1,1])

        self.Mirror_probability = ops.CoinFlip(probability=0.5)
        self.Rotation_probability = ops.Uniform(range = (-15,15))
        self.train_resize = ops.Resize(device='gpu', resize_x=256, resize_y=192)
        self.Normalize = ops.CropMirrorNormalize(device="gpu",
                                               dtype=types.FLOAT,
                                               output_layout=types.NCHW,
                                               crop_pos_x=0,
                                               crop_pos_y=0,
                                               mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.sequence_resize = ops.Resize(device='gpu', resize_x=416, resize_y=256, dtype=types.FLOAT)
        self.sequence_tran_FCHW = ops.Transpose(device="gpu", perm=[0, 3, 1, 2])
        self.sequence_Normalize = ops.Normalize(device="gpu",mean = 0.456 * 255,stddev = 0.224 * 255,dtype=types.FLOAT)

        self.val_resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)


    def define_graph(self):
        self.Scene_img = self.input_Scene_img()
        self.Scene_sequence = self.input_Scene_sequence()
        self.Label = self.intput_Label()
        if self.load_type == 'cam':
            self.index_list = self.input_index_list()
            index_list = self.index_list.gpu()

        if self.model_type == 'train':
            Scene_img = self.decode(self.Scene_img)
            Scene_sequence = self.Scene_sequence.gpu()

            Rotation_probability = self.Rotation_probability()
            Scene_img = fn.rotate(Scene_img, angle=Rotation_probability, fill_value=0)
            Scene_sequence = fn.rotate(Scene_sequence, angle=Rotation_probability, fill_value=0)
            Mirror_probability = self.Mirror_probability()
            Scene_img = fn.flip(Scene_img, horizontal=Mirror_probability)
            Scene_sequence = fn.flip(Scene_sequence, horizontal=Mirror_probability)

            Scene_img = self.Sence_img_resize(Scene_img)
            Scene_img = self.Normalize(Scene_img)
            Scene_img = self.crop(Scene_img)

            Scene_sequence = self.sequence_resize(Scene_sequence)
            Scene_sequence = self.sequence_tran_FCHW(Scene_sequence)
            Scene_sequence = self.crop(Scene_sequence)
        if self.model_type == 'val':
            Scene_img = self.decode(self.Scene_img)
            # Scene_img = self.Sence_img_resize(Scene_img)
            Scene_img = self.val_resize(Scene_img)
            Scene_img = self.Normalize(Scene_img)
            Scene_sequence = self.Scene_sequence.gpu()
            # Scene_sequence = self.sequence_resize(Scene_sequence)
            Scene_sequence = self.val_resize(Scene_sequence)
            Scene_sequence = self.sequence_tran_FCHW(Scene_sequence)
        Label = self.Label.gpu()
        if self.load_type == 'train':
            return (Scene_img, Scene_sequence, Label)
        elif self.load_type == 'cam':
            return (Scene_img, Scene_sequence, Label,index_list)


    def iter_setup(self):
        try:
            if self.load_type == 'train':
                Scene_img,Scene_sequence,Label  = self.iterator.next()
                self.feed_input(self.Scene_img, Scene_img)
                self.feed_input(self.Scene_sequence, Scene_sequence)
                self.feed_input(self.Label, Label)
            elif self.load_type == 'cam':
                Scene_img, Scene_sequence, Label, index_list = self.iterator.next()
                self.feed_input(self.Scene_img, Scene_img)
                self.feed_input(self.Scene_sequence, Scene_sequence)
                self.feed_input(self.Label, Label)
                self.feed_input(self.index_list, index_list)

        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class CustomDALIGenericIterator(DALIGenericIterator):
    def __init__(self, length,  pipelines,output_map,load_type='train', **argw):
        self._len = length # dataloader 的长度
        output_map = output_map
        self.load_type = load_type
        super().__init__(pipelines, output_map, **argw)

    def __next__(self):
        batch = super().__next__()
        return self.parse_batch(batch)

    def __len__(self):
        return self._len

    def parse_batch(self, batch):
        Scene_img = batch[0]['SGS']
        Scene_sequence = batch[0]['DLS']
        Label = batch[0]['Label']
        if self.load_type == 'cam':
            index_list = batch[0]['index_list']
            return {"SGS": Scene_img, "DLS": Scene_sequence,'Label':Label,'index_list':index_list}
        elif self.load_type == 'train':
            return {"SGS": Scene_img, "DLS": Scene_sequence,'Label':Label}

# if __name__ == '__main__':
#     Data_root_path = '../Dataset/'
#     train_inf = [json.loads(line) for line in open(Data_root_path + 'train.json')][0]
#     batch_size  = 64
#     num_threads = 12
#     load_type = 'cam'
#     train_eii = DataSource(batch_size=batch_size, Datainf=train_inf, Data_root_path= Data_root_path,shuffle=False,load_type=load_type,seq_number=4)
#     train_pipe = SourcePipeline(batch_size=batch_size, num_threads=num_threads, device_id=0, external_data=train_eii,
#                                 modeltype='train',load_type=load_type)
#     if load_type == 'train':
#         train_iter = CustomDALIGenericIterator(len(train_eii) / batch_size, pipelines=[train_pipe],
#                                                output_map=["SGS", 'DLS', 'Label'],
#                                                last_batch_padded=False,
#                                                size=len(train_eii),
#                                                last_batch_policy=LastBatchPolicy.PARTIAL,
#                                                auto_reset=True,
#                                                load_type=load_type)
#     elif load_type == 'cam':
#         train_iter = CustomDALIGenericIterator(len(train_eii) / batch_size, pipelines=[train_pipe],
#                                                output_map=["SGS", 'DLS', 'Label','index_list'],
#                                                last_batch_padded=False,
#                                                size=len(train_eii),
#                                                last_batch_policy=LastBatchPolicy.PARTIAL,
#                                                auto_reset=True,
#                                                load_type=load_type)
#     train_bar = tqdm(total=int(train_iter._len), iterable=train_iter, file=sys.stdout)
#     start_time = datetime.datetime.now()
#     for epochs in range(1):
#         for i, batch in enumerate(train_bar):
#             if load_type == 'train':
#                 Scene_img,Scene_sequence,Label = batch['SGS'],batch['DLS'],batch['Label']
#             elif load_type == 'cam':
#                 Scene_img, Scene_sequence, Label, index_list = batch['SGS'], batch['DLS'], batch['Label'], batch['index_list']
#                 print(index_list)
#             # Scene_img = Scene_sequence[0,0,:,:,:]
#             # Scene_img = Scene_img[0,  :, :, :]
#             # Scene_img = Scene_img.cpu().numpy()
#             # Scene_img = Scene_img.transpose(1,2, 0)
#             # plt.imshow(Scene_img)
#             # plt.show()
#     end_time = datetime.datetime.now()
#     elapsed_time = end_time - start_time
#     print('使用Dali循环的耗时为{}'.format(elapsed_time.total_seconds()))