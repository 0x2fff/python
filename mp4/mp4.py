import cv2
import numpy as np
import sys
from PIL import Image

class Reader(object):
    __video : cv2.VideoCapture
    __out_type = 'cv2'
    __step = 1
    __next_stop_flag = False

    def __init__(self, filepath, out_type='cv2'):
        self.__video = cv2.VideoCapture(filepath)
        self.__out_type = out_type

    def __del__(self):
        self.__video.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self.__next_stop_flag == True:
            self.__next_stop_flag = False
            raise StopIteration() 
        ret, frame = self.__video.read()
        if ret == False:
            raise StopIteration()
        if self.__step != 1:
            ret = self.seek_relative(self.__step - 1)
            if ret == False:
                self.__next_stop_flag = True
        return self.__convert_image(frame, self.__out_type)

    def reverse_iterator(self):
        self.set_step(-1)
        self.seek_absolute(self.get_frame_count() - 1)
        return self.__iter__()
        
    def set_step(self, number):
        self.__step = number

    def seek_absolute(self, number):
        return self.__video.set(cv2.CAP_PROP_POS_FRAMES, number)

    def seek_relative(self, number):
        current = self.get_position()
        next = current + number
        if next < 0:
            return False
        return self.seek_absolute(next)

    def skip_frame(self):
        return self.seek_relative(1)

    def get_image_data(self, number):
        current = self.get_position()
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, number)
        ret, frame = self.__video.read()
        self.seek_absolute(current)
        if ret:
            return self.__convert_image(frame, self.__out_type)

    def get_image_size(self):
        w = int(self.__video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def get_frame_count(self):
        return int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_position(self):
        return int(self.__video.get(cv2.CAP_PROP_POS_FRAMES))

    def get_fps(self):
        return self.__video.get(cv2.CAP_PROP_FPS)

    def get_fourcc(self):
        return int(self.__video.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('utf-8')

    def __convert_image(self, cv_image, out_type='cv2'):
        if out_type == 'cv2':
            return cv_image
        elif out_type == 'PIL':
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        elif out_type == 'numpy':
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            print("out_type \"{}\" is not supported".format(out_type),file=sys.stderr)
            raise ValueError()


class Writer(object):
    __video : cv2.VideoWriter

    def __init__(self, filepath:str, fourcc:str, fps:float, size:(int,int)):
        self.__video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), fps, size)

    def __del__(self):
        self.__video.release()
    
    def write(self, img):
        self.__video.write(img)


class Converter(object):
    fourcc = 'copy'
    def __select_fourcc(self, fourcc):
        if self.fourcc != 'copy':
            fourcc = self.fourcc
        return fourcc

    def downsampling(self, input_path, output_path, rate=0.5):
        r = Reader(input_path)
        fourcc = self.__select_fourcc(r.get_fourcc())
        w = Writer(output_path, fourcc, r.get_fps() * rate, r.get_image_size())
        r.set_step(1 / rate)
        for i in r:
            w.write(i)

    def reverse(self, input_path, output_path):
        r = Reader(input_path)
        fourcc = self.__select_fourcc(r.get_fourcc())
        w = Writer(output_path, fourcc, r.get_fps(), r.get_image_size())
        for i in r.reverse_iterator():
            w.write(i)

    def vstack(self, input_path:[], output_path):
        rlist = []
        size = (0, 0)
        for i in input_path:
            r = Reader(i)
            wh = r.get_image_size()
            size = (wh[0], size[1] + wh[1])
            rlist.append(r)
        
        fourcc = self.__select_fourcc(r.get_fourcc())
        w = Writer(output_path, fourcc, rlist[0].get_fps(), size)

        count = 0
        while True:
            dst = np.zeros((size[1], size[0], 3), np.uint8)
            pos = (0, 0)
            for r in rlist:
                src = r.get_image_data(count)
                if src is None:
                    return
                wh = r.get_image_size()
                dst[pos[1]:pos[1]+wh[1], pos[0]:pos[0]+wh[0]] = src
                pos = (pos[0], pos[1] + wh[1])
            w.write(dst)
            count = count + 1
