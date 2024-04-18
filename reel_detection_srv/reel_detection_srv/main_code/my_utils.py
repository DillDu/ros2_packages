import time
import random
import timeit
import os
import glob

colors_list = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 128, 0),     # Orange
    (128, 0, 255),     # Purple
    (0, 255, 128),     # Lime
    (255, 0, 128),     # Pink
    (128, 255, 0),     # Chartreuse
    (0, 128, 255),     # Azure
    (255, 128, 128),   # Salmon
    (128, 255, 128),   # Spring Green
    (128, 128, 255),   # Cornflower Blue
    (255, 128, 0),     # Tangerine
    (128, 0, 128),     # Indigo
    (0, 128, 128),     # Teal
    (128, 0, 0),       # Maroon
    (128, 128, 0),     # Olive
    (0, 128, 0),       # Green
    (255, 255, 128),   # Pale Yellow
    (255, 128, 255),   # Lavender
    (128, 255, 255),   # Light Cyan
    (255, 192, 128),   # Peach
    (192, 128, 255),   # Periwinkle
    (128, 255, 192),   # Seafoam Green
    (255, 128, 192),   # Rose
    (192, 255, 128),   # Lime Green
    (128, 192, 255),   # Sky Blue
]

def get_color(index, is_bgr = True, max_1 = False):
    color = colors_list[index%len(colors_list)]
    if max_1:
        color = (color[0]/255,color[1]/255, color[2]/255)
    if is_bgr:
        return (color[2],color[1],color[0])
    return color

def random_color(max_1=True):
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    if max_1:
        return (red/255, green/255, blue/255)
    
    return (red, green, blue)

def timer(func, *args):
    print("###### Start function: {}\n".format(func.__name__))
    start = time.time()
    ret = func(*args)
    end = time.time()
    print("###### Finish time: {}s".format(end - start))
    return ret

def time_func(func, *args, repeat = 1):
    """ time function with repeat num

    Args:
        func (function): test function
        *args: function arguments
        repeat (int, optional): repeat times. Defaults to 1.
    """
    t = timeit.Timer(lambda: func(*args))
    print(f">>>>>> Execute time: {t.timeit(repeat):.4f}s <<<<<<")
    
def get_file_paths(folder_path, *file_suffixes):
    file_paths = []
    if len(file_suffixes) == 0:
        file_paths.extend(glob.glob(f"{folder_path}/*"))
    else:
        for suffix in file_suffixes:
            file_paths.extend(glob.glob(f"{folder_path}/*.{suffix}"))
    file_paths = sorted(file_paths)
    
    return file_paths
        

if __name__ == '__main__':

    pass