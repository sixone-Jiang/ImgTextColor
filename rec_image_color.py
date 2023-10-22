from collections import namedtuple, Counter
from math import sqrt
import random
from PIL import Image
import numpy as np
import cv2

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(filename, n=2):
    img = Image.open(filename)
    img.thumbnail((20, 20))
    w, h = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1) # Alice : change here for topk color
    #print(clusters)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def colorz_cv2(cv2_image, n=2):

    img = Image.fromarray(cv2_image)
    img.thumbnail((20, 20))
    w, h = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1) # Alice : change here for topk color
    #print(clusters)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    #min_diff = 100
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break
    
    # Alice change, delete clusters with most points
    max_points = 0
    max_idx = 0
    for i in range(k):
        print(len(clusters[i].points))
        if len(clusters[i].points) > max_points:
            max_points = len(clusters[i].points)
            max_idx = i
    del clusters[max_idx]
    return clusters

def show_image_color(result_color):
    print(result_color)
    img = Image.new('RGB', (50, 50), result_color)
    # save image
    img.save('./upload_image/result.jpg')

def rec_an_image_color(filename, show_demo=False):
    colors = colorz(filename)
    result_color = ''
    for color in colors:
        result_color = color

    if show_demo:
        show_image_color(result_color)

    return result_color

def rec_an_image_color_cv2(cv2_image, show_demo=False):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    colors = colorz_cv2(cv2_image, n=2)
    result_color = ''
    for color in colors:
        result_color = color
        print(color)

    if show_demo:
        show_image_color(result_color)

    return result_color

def make_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 黑白颠倒
    #gray = cv2.bitwise_not(gray)
    # 二值化
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    return thresh

def pre_work(cv2_image):
    thresh = make_gray(cv2_image)
    # when loc_pix of thresh is 255, set loc_pix of cv2_image to 255
    h, w = thresh.shape[:2]
    for i in range(h):
        for j in range(w):
            if thresh[i][j] == 255:
                cv2_image[i][j] = 255
    return cv2_image

def use_grab_cut(cv2_image):
    mask = np.zeros(cv2_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # rect is like the mask box
    width = cv2_image.shape[1]
    height = cv2_image.shape[0]
    rect = (int(0.05*width), int(0.05*height), int(0.9*width), int(0.9*height))
    cv2.grabCut(cv2_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # let background is white, foreground is black
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    cv2_image = cv2_image*mask2[:,:,np.newaxis]
    background = np.ones_like(cv2_image)*255
    background = background * (1 - mask2[:,:,np.newaxis])
    cv2_image = cv2_image + background
    return cv2_image

# make all image pix color change to the closest 8 color`s max_num color
def max_pooling(image, pool_size=(6, 6)):
    pooled_image = cv2.resize(image, (0, 0), fx=1/pool_size[1], fy=1/pool_size[0], interpolation=cv2.INTER_LINEAR)
    return pooled_image

# sharpening
def sharpening(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def contraharmonic_mean(img, size=(3,3), Q=0.5):
    num = np.power(img, Q + 1)
    denom = np.power(img, Q)
    kernel = np.full(size, 1.0)
    result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
    return result

# Canny edge detection
def canny_edge_detection(image):
    # fix bug _src.depth() == CV_8U
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 100, 200)
    return edges

def find_and_draw_contours(image, output_image_path, min_contour_area=10):

    # 查找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小轮廓
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # 创建一个全白的图像
    height, width = image.shape
    white_background = np.ones((height, width), dtype=np.uint8) * 255

    # 在白色背景上绘制黑色轮廓
    cv2.drawContours(white_background, filtered_contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    # 保存输出图像
    cv2.imwrite(output_image_path, white_background)

    return white_background

def cluster_pixels(image_path, k=5):
    # 读取输入图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 将图像数据重塑为像素点列表
    pixels = image.reshape(-1, 3).astype(np.float32)
    print('pixels', pixels.shape)

    # 使用K均值聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # change centers to uint8
    centers = np.uint8(centers)

    print('centers', centers)

    segmented_image = centers[labels.flatten()].reshape(height, width, 3).astype(np.uint8)
    image_like_label = labels.reshape(height, width).astype(np.uint8)
    return segmented_image, image_like_label, centers

def cluster_list_pixels(pixels, k=1):
    # 将图像数据重塑为像素点列表
    pixels = pixels.reshape(-1, 3).astype(np.float32)
    print('pixels', pixels.shape)

    # 使用K均值聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    return centers[0]

def extract_pixels(image):
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 定义要提取的像素块的坐标
    regions = [
        (0, 0, 2, 2),                 # 左上角 2x2
        (0, width - 2, 2, width),     # 右上角 2x2
        (height - 2, 0, height, 2),   # 左下角 2x2
        (height - 2, width - 2, height, width),  # 右下角 2x2
        (0, 0, 1, width),             # 上边线 1x1
        (0, width - 1, 1, width),     # 下边线 1x1
        (0, 0, height, 1),            # 左边线 1x1
        (height - 1, 0, height, 1)    # 右边线 1x1
    ]

    pixels = []

    for region in regions:
        x1, y1, x2, y2 = region
        block = image[x1:x2, y1:y2]
        pixels.extend(block.reshape(-1, 3))

    return np.array(pixels)

def find_most_common_color(image):
    # 提取像素
    pixels = extract_pixels(image)

    # 使用Counter来计算颜色频率
    color_counter = Counter(map(tuple, pixels))

    # 获取出现最频繁的颜色
    most_common_color = color_counter.most_common(1)[0][0]

    return most_common_color

def get_text_color(segmented_image, image_like_label, centers):
    most_common_color = find_most_common_color(segmented_image)
    print('most_common_color', most_common_color)
    background_color_index = centers.tolist().index(list(most_common_color))
    print('background_color_index', background_color_index)
    if background_color_index == 0:
        front_asume_color = centers[1]
    else:
        front_asume_color = centers[0]

    return background_color_index, front_asume_color

def convert_bgr2hex(bgr):
    b, g, r = bgr
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# use original image to get text color
def get_text_color_upper(bef_image, background_color_index, image_like_label):

    assert bef_image.shape[:2] == image_like_label.shape
    pixs_lists = []
    for i in range(bef_image.shape[0]):
        for j in range(bef_image.shape[1]):
            if image_like_label[i][j] != background_color_index:
                pixs_lists.append(bef_image[i][j])
    
    upper_front_asume_color = cluster_list_pixels(np.array(pixs_lists), k=1)
    return upper_front_asume_color

def rec_an_image_color_cv2_v2(cv2_image, show_demo=False, use_upper_algrithm=False):
    #cv2_image = contraharmonic_mean(cv2_image) # in some case, this will make the bad result
    cv2.imwrite('./upload_image/pre_work.png', cv2_image)
    segmented_image, image_like_label, centers = cluster_pixels('./upload_image/pre_work.png', k=2)
    background_color_index, front_asume_color = get_text_color(segmented_image, image_like_label, centers)
    hex_color = convert_bgr2hex(front_asume_color)
    cv2.imwrite('./upload_image/seg_image.png', segmented_image)
    if use_upper_algrithm:
        bef_image = cv2.imread('./upload_image/pre_work.png')
        upper_front_asume_color = get_text_color_upper(bef_image, background_color_index, image_like_label)
        hex_color = convert_bgr2hex(upper_front_asume_color)
    if show_demo:
        show_image_color(hex_color)
    return hex_color

if __name__=="__main__":
    # read image and get colors
    filename = './test_image/test.jpg' # change here for your own image
    #colors = colorz(filename)
    cv2_image = cv2.imread(filename)

    #cv2_image = pre_work(cv2_image)
    #cv2_image = use_grab_cut(cv2_image)
    cv2_image = contraharmonic_mean(cv2_image)
    #cv2_image = use_grab_cut(cv2_image)
    #cv2.imwrite('./upload_image/pre_work.jpg', cv2_image)
    #cv2_image = max_pooling(cv2_image, pool_size=(3, 3))
    cv2.imwrite('./upload_image/pre_work.jpg', cv2_image)
    cv2_image = cv2.imread('./upload_image/pre_work.jpg')
    cv2_image = canny_edge_detection(cv2_image)

    cv2.imwrite('./upload_image/canny.jpg', cv2_image)
    # cv2_image = find_and_draw_contours(cv2_image, './upload_image/contours.jpg')
    #cv2_image = use_grab_cut(cv2_image)
    #cv2.imwrite('./upload_image/pre_work.jpg', cv2_image)
    #rec_an_image_color_cv2(cv2_image, show_demo=False)
    segmented_image, image_like_label, centers = cluster_pixels('./upload_image/pre_work.jpg', k=2)
    bef_image = cv2.imread('./upload_image/pre_work.jpg')
    background_color_index, front_asume_color = get_text_color(segmented_image, image_like_label, centers)
    print('front_asume_color', convert_bgr2hex(front_asume_color))
    upper_front_asume_color = get_text_color_upper(bef_image, background_color_index, image_like_label)
    print('upper_front_asume_color', convert_bgr2hex(upper_front_asume_color))
    cv2.imwrite('./upload_image/seg_image.png', segmented_image)
    