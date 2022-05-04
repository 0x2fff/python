import mp4
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def main():
    def test_video1():
        w = mp4.Writer("test1.mp4", 'mp4v', 20, (1920,1080))
        for i in range(100):
            img = np.zeros((1080, 1920, 3), np.uint8)
            img = cv2.putText(img, "{:02d}".format(i), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
            w.write(img)

    def test_video2():
        r = mp4.Reader("test1.mp4")
        w = mp4.Writer("test2.mp4", r.get_fourcc(), r.get_fps(), r.get_image_size())
        for i in r:
            img = cv2.bitwise_not(i)
            w.write(img)

    test_video1()
    test_video2()
    c = mp4.Converter()
    c.fourcc = 'mp4v'
    with ThreadPoolExecutor() as executor:
        executor.submit(c.downsampling, "test1.mp4", "test3_downsampling.mp4")
        executor.submit(c.reverse, "test1.mp4", "test4_reverse.mp4")
        executor.submit(c.vstack, ["test1.mp4","test2.mp4"],"test5_vstack.mp4")

if __name__ == '__main__':
    main()
