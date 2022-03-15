
from distutils.command.config import config
from turtle import right
import torch
def iou(self, box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2), # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N, M, 2]
    wh[wh < 0] = 0 # 两个box没有重叠的区域
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) # (N, )
    area2 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) # (M, )

    area1 = area1.unsqueeze(1).expand(N, M) # (N, M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N, M)
    
    iou = inter / (area1 + area2 - inter)
    return iou



def iou2(rect1, rect2):
    '''
    rect: [xmin1, ymin1, xmax1, ymax1]
    '''
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    left = max(xmin1, xmin2)
    right = min(xmax1, xmax2)
    top = max(ymin1, ymin2)
    bottom = min(ymax1, ymax2)

    if left >= right or top >= bottom:
        return 0
    
    intersection = (right - left) * (bottom - top)

    return intersection / (s1 + s2 - intersection)



#------------------------------------
# nms算法流程
# 1. 先将所有候选框的置信度排序，因为我们最终是要最大的
# 2. 将置信度最大的加入到最终的返回值中
# 3. 将其他的候选框和当前的最大的那个计算IOU
# 4. 如果IOU大于一个阈值， 则删除（说明和置信度大的那个是重叠的）
# 5. 将剩下的框重复以上过程
#------------------------------------

import cv2
import numpy as np

boxes=np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]])

def nms(bboxes, threshold):
    # 计算所有候选框的面积，为IOU作准备
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # areas.shape [6,]
    order = np.argsort(scores) # argsort函数返回经过排序之后，有序数组各个位置上是原数组的哪个元素的索引，这就意味着我们不需要对bbox进行排序，只需要对置信度进行排序
    keep = [] # 返回值

    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中，对应1，2步
        index = order[-1] # 最大置信度框的index
        keep.append(index)
       
       
        # 对应第3步 计算其他框和当前选定的框的iou，因为这里数据类型是np，所以一次是计算的多个
        left = np.maximum(x1[index], x1[order[:-1]]) # array的索引可以是一个array，返回一个对应索引的array
        right = np.minimum(x2[index], x2[order[:-1]])

        top = np.maximum(y1[index], y1[order[:-1]])
        bottom = np.minimum(y2[index], y2[order[:-1]])

        in_w = np.maximum(0.0, right - left + 1)
        in_h = np.maximum(0.0, bottom - top + 1)

        inner = in_h * in_w

        ratio = inner / (areas[index] + areas[order[:-1]] - inner)
        keep_idx = np.where(ratio < threshold) # keep_idx 里面对应的就是<thr的索引
        order = order[keep_idx] # 将所有<thr的索引取出来

    return keep
    
import matplotlib.pyplot as plt
def plot_bbox(dets, c='k'):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    plt.plot([x1,x2], [y1,y1], c)
    plt.plot([x1,x1], [y1,y2], c)
    plt.plot([x1,x2], [y2,y2], c)
    plt.plot([x2,x2], [y1,y2], c)
    plt.title(" nms")





def my_nms(bboxes, threshold):
    x1, y1, x2, y2, confidence = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]

    area = (x2 - x1) * (y2 - y1)
    indices = confidence.argsort()[::-1]
    keep = []
    while indices.size > 0:
        idx_self, idx_other = indices[0], indices[1:]
        keep.append(idx_self)

        xx1, yy1 = np.maximum(x1[idx_self], x1[idx_other]), np.maximum(y1[idx_self], y1[idx_other])
        xx2, yy2 = np.minimum(x2[idx_self], x2[idx_other]), np.minimum(y2[idx_self], y2[idx_other])

        w, h = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)

        iou = w * h / (area[idx_self] + area[idx_other] - w * h)
        tmp = np.where(iou < threshold)[0]
        indices = indices[tmp + 1]
    return keep

def my_nms2(bboxs, threshold):
    x1, y1, x2, y2, score = bboxs[:, 0], bboxs[:, 1], bboxs[:, 2], bboxs[:, 3], bboxs[:, 4]

    area = (x2 - x1) * (y2 - y1)
    indice = score.argsort()[::-1]
    keep = []
    while indice.size > 0:
        idx_keep = indice[0]
        keep.append(idx_keep)

        xx1, yy1 = np.maximum(x1[idx_keep], x1[1:]), np.maximum(y1[idx_keep], y1[1:])
        xx2, yy2 = np.minimum(x2[idx_keep], x2[1:]), np.minimum(y2[idx_keep], y2[1:])
        w, h = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)

        intersection = w * h
        the_iou = intersection / (area[idx_keep] + area[1:] - intersection)

        tmp = np.where(the_iou <= threshold)[0]
        indice = indice[tmp + 1]
    return keep


plt.figure(1)
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
 
plt.sca(ax1)
plot_bbox(boxes,'k')   # before nms
 
keep = my_nms(boxes,0.7)
plt.sca(ax2)
plot_bbox(boxes[keep], 'r')# after nm
plt.show()
    













        