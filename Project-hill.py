import numpy as np
import cv2
import math
import time
import random
t1 = time.time()
#####################################
n_colors = 3
alpha = 40
img_path_input = 'img\Im001_1.tif'
img_path_output = ''
n_iterations = 50
n_initial_points = 2000
n_points = 10
stop_stuck = 7
alpha_dec = 0.9
#####################################
transition = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [0, -1, -1], [0, -1, 0], [0, 0, -1], [1, 0, -1], [0, 1, -1], [1, -1, 0], [-1, 1, 0], [0, -1, 1], [-1, 0, 1], [0, 0, 0]]
transition = np.asarray(transition)
# Converting image to numpy array
imgage = np.asarray(cv2.imread(img_path_input, 1))
img = cv2.resize(imgage, dsize=(15, 15), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def PSNR(original, compressed): 
#     mse = np.mean((original - compressed) ** 2) 
#     if(mse == 0):  # MSE is zero means no noise is present in the signal . 
#                   # Therefore PSNR have no importance. 
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * math.log10(max_pixel / mse) 
#     return psnr 

def make_image(point):
    fitness = 0
    new_img = np.copy(img)
    z = 0
    for f in range(len(new_img)):
        for i in range (len(new_img[f])):
            min_ = 10000
            for j in point:
                if (np.linalg.norm(new_img[f, i] - j) < min_):
                    min_ = np.linalg.norm(new_img[f, i] - j)
                    x = j
            new_img[f, i] = x
            z = z + 1
            fitness += min_
    return [new_img, fitness]

def fitness(point):
    # t1 = time.time()
    fitness = 0
    for x in img:
        for i in x:
            min_ = 10000000000
            for j in point:
                min_ = min(np.linalg.norm(i - j), min_)
            fitness += min_
    # t2 = time.time()
    # print(t2 - t1)
    return fitness

# each time we change one color among three, so we make (3 ^ n_colors ) * 3 neighbours.
def Best_neighbour(point, transition):
    t1 = time.time()
    global alpha
    transition = transition * alpha
    # t1 = time.time()
    temp_point = []
    for i in transition:
        for z in range(n_colors):
            for j in range(n_colors):
                temp = (i if z == j else np.asarray([0, 0, 0]))  + point[j]
                temp_point = np.concatenate((temp_point, temp), axis = 0)
    best = 1000000000
    worst = 0
    temp_point = np.reshape(temp_point, (3*3*3*n_colors,n_colors,3))
    for i in range(3*3*3*n_colors):
        done = 0
        fit = fitness(temp_point[i])
        if (fit < best):
            best = fit
            colors = temp_point[i]
        if ((i == 3*3*3*n_colors - 1) and (fit == best)):
            done = 1
    t2 = time.time()
    # print(t2 - t1)
    return [best, colors, done]
    # print(worst)
# print(Best_neighbour(np.asarray([[156, 160, 155], [167, 168, 164], [165, 166, 162]]), transition))
# print(fitness([[140, 128, 150], [175, 173, 175], [133, 31, 75]]) / np.size(img) * 3)
final_points = {}
points_dic = {}
#to move elements after finding a better neighbour in.
points_dic_move = {}
points = []
points = np.asarray(points)


for i in range(n_colors * n_initial_points):
    x = (random.random() * imgage.shape[0]) // 2 * 2
    x = int(x)
    y = (random.random() * imgage.shape[1]) // 2 * 2
    y = int(y)
    points = np.concatenate((points, imgage[x, y]), axis = 0)
points = np.reshape(points, (n_initial_points, n_colors, 3))
for i in points:
    points_dic[fitness(i)] = i

new_points = {}
for i in sorted(points_dic.keys())[:n_points]:
    new_points[i] = points_dic[i]
points_dic = new_points

#now we have n_points points that are selected for hillclimbing search
for i in range(n_iterations):
    while (points_dic):
        l1 = len(final_points)
        l2 = len(points_dic)
        j = list(points_dic.keys())[0]
        x = Best_neighbour(points_dic[j], transition)
        if(x[2] == 1 and transition[0, 0] * alpha < stop_stuck):   #we stuck in one hill
            final_points[j] = points_dic[j]
            points_dic.pop(j)
        else: #we have a better neighbour or we stuck and don't want to stop.
            points_dic.pop(j)
            points_dic_move[x[0]] = x[1]
        if (transition[0, 0] * alpha < stop_stuck and alpha_dec <= 0.92):
            alpha_dec = alpha_dec + 0.07
    print(i + 1, " : ", len(final_points))
    points_dic = points_dic_move
    points_dic_move = {}
    if (transition[0, 0] * alpha > 1):
        alpha = alpha * alpha_dec
    print("move =", transition[0, 0] * alpha)
img = np.asarray(cv2.imread(img_path_input, 1))
final_points.update(points_dic)
print("img001")
print(final_points[sorted(final_points.keys())[0]])

# final_points[fitness([[140.17560261, 128.03699817, 151.87967467], [176.64421492, 175.30007546, 175.93095106], [131.73155149, 25.71763357, 87.30391448]])]= [[140.17560261, 128.03699817, 151.87967467], [176.64421492, 175.30007546, 175.93095106], [131.73155149, 25.71763357, 87.30391448]]


image = make_image(final_points[sorted(final_points.keys())[0]])
fitness = image[1]
image = image[0]
img1 = image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print("PSNR :", cv2.PSNR(gray, gray1))
print(cv2.PSNR(gray, gray1))
# img1 = np.asarray(img1)
# print(img1)
print("time :", (time.time() - t1) / 60)
print("fitness :", (fitness / np.size(img)) * 3)
cv2.imwrite(img_path_output, img1)
# cv2.imshow('img1', img1)
# cv2.imshow('img', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





