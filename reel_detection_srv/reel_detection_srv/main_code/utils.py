import time

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

def get_color(index, is_bgr = False):
    color = colors_list[index%30]
    if is_bgr:
        return (color[2],color[1],color[0])
    return color

def timer(func, *args):
    print("###### Start function: {}\n".format(func.__name__))
    start = time.time()
    ret = func(*args)
    end = time.time()
    print("###### Finish time: {}s".format(end - start))
    return ret