import time
import paddle
import paddle.nn as nn
import numpy as np


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = paddle.zeros([len(boxes)])
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    sortIds = paddle.argsort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def convert2cpu(gpu_matrix):
    float_32_g = gpu_matrix.astype('float32')
    return float_32_g.cpu()

def convert2cpu_long(gpu_matrix):
    int_64_g = gpu_matrix.astype('int64')
    return int_64_g.cpu()
    
def get_region_boxes(output, conf_thresh=0.005, num_classes=24, anchors=[0.70458, 1.18803, 1.26654, 2.55121, 1.59382, 4.08321, 2.30548, 4.94180, 3.52332, 5.91979], num_anchors=5, only_objectness=1, validation=False):
    anchor_step = len(anchors)//num_anchors
    if output.dim() == 3 :
        output = output.unsqueeze(0)
    batch = output.shape[0]
    assert(output.shape[1] == (5+num_classes)*num_anchors)
    h = output.shape[2]
    w = output.shape[3]
    t0 = time.time()
    all_boxes = []
    output = paddle.reshape(output, [batch*num_anchors, 5+num_classes, h*w])
    output = paddle.transpose(output, (1, 0, 2))
    output = paddle.reshape(output, [5+num_classes, batch*num_anchors*h*w])

    grid_x = paddle.linspace(0, w-1, w)
    grid_x = paddle.tile(grid_x, [h, 1])
    grid_x = paddle.tile(grid_x, [batch*num_anchors, 1, 1])
    grid_x = paddle.reshape(grid_x, [batch*num_anchors*h*w]).cuda()

    grid_y = paddle.linspace(0, h-1, h)
    grid_y = paddle.tile(grid_y, [w, 1]).t()
    grid_y = paddle.tile(grid_y, [batch*num_anchors, 1, 1])
    grid_y = paddle.reshape(grid_y, [batch*num_anchors*h*w]).cuda() 

    sigmoid = nn.Sigmoid()
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = paddle.to_tensor(anchors)
    anchor_w = paddle.reshape(anchor_w, [num_anchors, anchor_step])
    anchor_w = paddle.index_select(anchor_w, index = paddle.to_tensor(np.array([0]).astype('int32')), axis = 1)

    anchor_h = paddle.to_tensor(anchors)
    anchor_h = paddle.reshape(anchor_h, [num_anchors, anchor_step])
    anchor_h = paddle.index_select(anchor_h, index = paddle.to_tensor(np.array([1]).astype('int32')), axis = 1)

    anchor_w = paddle.tile(anchor_w, [batch, 1])
    anchor_w = paddle.tile(anchor_w, [1, 1, h*w])
    anchor_w = paddle.reshape(anchor_w, [batch*num_anchors*h*w]).cuda()

    anchor_h = paddle.tile(anchor_h, [batch, 1])
    anchor_h = paddle.tile(anchor_h, [1, 1, h*w])
    anchor_h = paddle.reshape(anchor_h, [batch*num_anchors*h*w]).cuda()    

    ws = paddle.exp(output[2]) * anchor_w
    hs = paddle.exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])

    cls_confs = paddle.to_tensor(output[5:5+num_classes],  stop_gradient=True)
    cls_confs = paddle.transpose(cls_confs, [1, 0])
    s = nn.Softmax()
    cls_confs = paddle.to_tensor(s(cls_confs))

    cls_max_confs = paddle.max(cls_confs, axis = 1)
    cls_max_ids = paddle.argmax(cls_confs, axis = 1)

    cls_max_confs = paddle.reshape(cls_max_confs, [-1])
    cls_max_ids = paddle.reshape(cls_max_ids, [-1])

    t1 = time.time()

    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.reshape([-1, num_classes]))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0]-box1[2]/2.0), float(box2[0]-box2[2]/2.0))
        Mx = max(float(box1[0]+box1[2]/2.0), float(box2[0]+box2[2]/2.0))
        my = min(float(box1[1]-box1[3]/2.0), float(box2[1]-box2[3]/2.0))
        My = max(float(box1[1]+box1[3]/2.0), float(box2[1]+box2[3]/2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea
