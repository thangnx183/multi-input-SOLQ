import json
import numpy as np
from pathlib import Path
import cv2
from pair_solq_model import Pair_SOLQ_model
import torch
from util import box_ops
from scipy.optimize import linear_sum_assignment
import os
from glob import glob
import time as TIME

CATEGORIES = [
    'sli+rf', 'sli+lf', 'tyre+rf', 'tyre+rb',
    'tyre+lf', 'tyre+lb', 'alloy_wheel+rf', 'alloy_wheel+rb',
    'alloy_wheel+lf', 'alloy_wheel+lb', 'hli+rf',
    'hli+lf', 'hood+f', 'fwi+f', 'flp+f',
    'door+rf', 'door+rb', 'door+lf', 'door+lb', 'mirror+rf', 'mirror+lf',
    'handle', 'qpa+rb', 'qpa+lb', 'fender+rf',
    'fender+lf', 'grille+f', 'fbu+f', 'rocker_panel+r',
    'rocker_panel+l', 'rbu+b', 'pillar', 'pillar', 'roof',
    'blp+b', 'window+r', 'window+l', 'rwi+b',
    'tail_gate+b', 'tli+rb', 'tli+lb',
    'fbe+rf', 'fbe+lf', 'fli+rf',
    'fli+lf', 'fuel_tank_door', 'lli+lb',
    'lli+rb', 'exhaust'
]
path = Path('coco_data/carpart-lr')

# data = json.load(open(path/'annotations_20222'/'valid.json'))
model = Pair_SOLQ_model(
        'exps-optimal-scheduler-3-e2e-opt-mask/phase-4-ema/multi-scale-optimal-scheduler-e2e-mask-ema.pth', 0.3, 'cuda:0')

# checkpoints/video-cp-side-fast-solq-48.pth
# tp :  39711  fp :  19643  fn :  13185 precision :  0.6690534757556357  recall :  0.7507372958257713  f1 : 0.7075456570150918
# exps-optimal-scheduler/cp-side-50-cates-fdct-phase-4/checkpoint.pth
# tp :  43873  fp :  14778  fn :  9023 precision :  0.748034986615744  recall :  0.8294199939503932  f1 : 0.7866280581274654

def vis(image, result,fp):
    draw_image = image.copy()
    # print('1')
    # draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)

    for idx,(b, l, c, m) in enumerate(zip(result['bboxes'], result['labels'], result['scores'], result['masks'])):
        if idx not in fp:
            clr = [255, 255, 0]
        else:
            clr = [0,255,255]
        # print(idx,l, c)
        # if CATEGORIES[l] == 'hli+lf':
        #     clr = [255,255,0]
        # elif CATEGORIES[l] == 'fbu+f':
        #     clr = [2550,0,255]

        # print(CATEGORIES[l],c)
        b = list(map(int, b))
        cv2.rectangle(draw_image, tuple(b[:2]), tuple(
            b[2:]), (102, 220, 225), thickness=2)

        if isinstance(l, str):
            cv2.putText(draw_image, l+'|'+str(round(c, 2)),
                        (b[0]//2+b[2]//2, b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 1)
        else:
            cv2.putText(draw_image, CATEGORIES[l]+'|'+str(round(
                c, 2)), (b[0]//2+b[2]//2, b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clr, 1)

        # print(m.shape)
        m = np.array(m).astype(np.uint8)
        _,cons, _ = cv2.findContours(
            m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_image = cv2.drawContours(draw_image, cons, -1, clr, 2)

    return draw_image


def matching(cost, pad_value):
    h, w = cost.shape
    if h != w:
        cost = cost.numpy()
        cost = np.pad(cost, [(0, (h < w)*(w-h)), (0, (h > w)*(h-w))],
                      'constant', constant_values=pad_value)
        cost = torch.tensor(cost)

    row_ind, col_ind = linear_sum_assignment(-cost)

    return row_ind, col_ind


def evaluate_result(gt_boxes, gt_labels, result):
    # print('pred  : ',[CATEGORIES[i] for i in result['labels']])
    # print('gt : ',[CATEGORIES[i] for i in gt_labels])

    pred_boxes = torch.as_tensor(result['bboxes'])

    gt_boxes = box_ops.box_xywh_to_xyxy(torch.as_tensor(gt_boxes))
    # print('gt ', gt_boxes)

    if len(pred_boxes) == 0:
        return [], [], [i for i in range(len(gt_labels))]
    ious, _ = box_ops.box_iou(pred_boxes, gt_boxes)

    gt_labels = torch.as_tensor(gt_labels)
    # print(gt_labels, result['labels'])
    match_label = torch.stack(
        [gt_labels == i for i in result['labels']]).type(torch.LongTensor)

    # print(match_label)
    # print(ious)
    cost = ious+match_label
    h, w = cost.shape
    row_ind, col_ind = matching(cost, pad_value=1)
    # print(ious, row_ind, col_ind)

    tp, fp, fn = [], [], []

    for r, c in zip(row_ind, col_ind):
        # FP
        if c >= w:
            fp.append(r)
            # print('fp : ',CATEGORIES[result['labels'][r]])
            continue

        # FN
        if r >= h:
            fn.append(c)
            # print('fn : ',CATEGORIES[gt_labels[c]])
            continue

        if ious[r, c] > 0.2 and match_label[r, c] == 1:  # TP
            tp.append(r)
        else:  # FP
            fp.append(r)
            fn.append(c)
            # print('fp : ',CATEGORIES[result['labels'][r]],CATEGORIES[gt_labels[c]])

    return tp, fp, fn


def inference(ref_path,transition_path,zoomin_path):
    # ref_path = 'IMG_1415/images/ref_time_7.216753.jpg'

    # transition_path = 'IMG_1415/images/transition_time_7.216753.jpg'
    transition_name = os.path.basename(transition_path)

    # zoomin_path = 'IMG_1415/images/zoomin_time_reftime_7.216753_9.146154.jpg'
    image_name = os.path.basename(zoomin_path)
    print('name : ',image_name)

    input_folder = os.path.dirname(zoomin_path)
    output_transition_folder = input_folder.replace('images','transition')
    output_original_folder = input_folder.replace('images','original')

    os.makedirs(output_transition_folder,exist_ok=True)
    os.makedirs(output_original_folder,exist_ok=True)

    ref = cv2.imread(ref_path)
    transition = cv2.imread(transition_path)
    zoomin2 = cv2.imread(zoomin_path)


    out1, cache_memory1 = model.inference_filenames(transition,ref,confident=0.2)
    out2_with_cache, _ = model.inference_filename_with_cache(zoomin2,cache_memory1,confident=0.25)
    filter_id = [i in out1['labels'] for i in out2_with_cache['labels']]
    out2_with_cache['bboxes'] = np.array(out2_with_cache['bboxes'])[filter_id].tolist()
    out2_with_cache['masks'] = np.array(out2_with_cache['masks'])[filter_id].tolist()
    out2_with_cache['labels'] = np.array(out2_with_cache['labels'])[filter_id].tolist()
    out2_with_cache['scores'] = np.array(out2_with_cache['scores'])[filter_id].tolist()

    # out_image = vis(transition, out1,[])
    # cv2.imwrite(os.path.join(output_transition_folder,transition_name),out_image)    
    # print('-'*50)
    # print('transition : ',out1['labels'], ' | ', out1['scores'])
    # print(out1['bboxes'])

    out2_with_ref_nms, _ = model.inference_filenames(zoomin2,ref)
    out2_new = ensemble(out2_with_ref_nms,out2_with_cache)
    
    # print('-'*50)
    # print('zoomin  with cache transition: ',out2_with_cache['labels'], ' | ', out2_with_cache['scores'])
    # print('zoomin  with cache transition after nms: ',out2_new['labels'], ' | ', out2_new['scores'])
    # print(out2_with_cache['bboxes'])
    # out_image = vis(zoomin2, out2_new,[])
    # cv2.imwrite(os.path.join(output_transition_folder,image_name),out_image)
    
    
    
    # out2_old, _ = model.inference_filenames_old(zoomin2,ref)
    # out_image = vis(zoomin2, out2_old,[])
    # cv2.imwrite(os.path.join(output_original_folder,image_name),out_image)
    
    # print('-'*50)
    # print('zoomin  with origin ref: ',out2_old['labels'], ' | ', out2_old['scores'])
    # print(out2_old['bboxes'])
    # print('*'*200)

    

def ensemble(out1,out2):
    if len(out2['bboxes']) == 0 :
        return out1
    boxes1 = torch.as_tensor(out1['bboxes'])
    boxes2 = torch.as_tensor(out2['bboxes'])
    # print(boxes1,boxes2)
    ious,_ = box_ops.box_iou(boxes1,boxes2)
    # print(ious)
    row_ind, col_ind = matching(ious,pad_value=1)
    h,w = ious.shape
    for r,c in zip(row_ind,col_ind):
        if r >= h or c >= w:
           continue
        if ious[r,c] > 0.8 and not (out1['labels'][r] == 'grille+f' and out2['labels'][c] == 'fbu_front_bumper+f'): 
            # print(out1['labels'][r], out2['labels'][c])
            out1['labels'][r] = out2['labels'][c]
    
    return out1
    
def inference_v2(ref_path,transition_path,zoomin_path):
    transition_name = os.path.basename(transition_path)

    image_name = os.path.basename(zoomin_path)
    # print('name : ',image_name)

    input_folder = os.path.dirname(zoomin_path)
    output_transition_folder = input_folder.replace('images','transition_v2')
    output_original_folder = input_folder.replace('images','original')

    os.makedirs(output_transition_folder,exist_ok=True)
    os.makedirs(output_original_folder,exist_ok=True)

    ref = cv2.imread(ref_path)
    transition = cv2.imread(transition_path)
    zoomin = cv2.imread(zoomin_path)
    
    
    out_transition, ref_memory, ref_memory_add_pos, transition_memory = model.inference_without_transition(
        transition,ref.copy(),0.3)
    
    out_image = vis(transition, out_transition,[])
    cv2.imwrite(os.path.join(output_transition_folder,transition_name),out_image) 
    
    out_zoomin = model.inference_with_transition(zoomin.copy(),ref_memory,ref_memory_add_pos,transition_memory,out_transition,0.25)
    
    out_image = vis(zoomin, out_zoomin,[])
    cv2.imwrite(os.path.join(output_transition_folder,image_name),out_image)


    # out1, cache_memory1 = model.inference_filenames(transition,ref,confident=0.2)
    # out2_with_cache, _ = model.inference_filename_with_cache(zoomin2,cache_memory1,confident=0.25)
    # filter_id = [i in out1['labels'] for i in out2_with_cache['labels']]
    # out2_with_cache['bboxes'] = np.array(out2_with_cache['bboxes'])[filter_id].tolist()
    # out2_with_cache['masks'] = np.array(out2_with_cache['masks'])[filter_id].tolist()
    # out2_with_cache['labels'] = np.array(out2_with_cache['labels'])[filter_id].tolist()
    # out2_with_cache['scores'] = np.array(out2_with_cache['scores'])[filter_id].tolist()

    # out_image = vis(transition, out1,[])
    # cv2.imwrite(os.path.join(output_transition_folder,transition_name),out_image)    
    # print('-'*50)
    # print('transition : ',out1['labels'], ' | ', out1['scores'])
    # print(out1['bboxes'])

    # out2_with_ref_nms, _ = model.inference_filenames(zoomin2,ref)
    # out2_new = ensemble(out2_with_ref_nms,out2_with_cache)
    
    # print('-'*50)
    # print('zoomin  with cache transition: ',out2_new['labels'], ' | ', out2_new['scores'])
    # print(out2_new['bboxes'])
    # out_image = vis(zoomin2, out2_new,[])
    # cv2.imwrite(os.path.join(output_transition_folder,image_name),out_image)
    
    
    
    # out2_old, _ = model.inference_filenames_old(zoomin2,ref)
    # out_image = vis(zoomin2, out2_old,[])
    # cv2.imwrite(os.path.join(output_original_folder,image_name),out_image)
    # print('-'*50)
    # print('zoomin  with origin ref: ',out2_old['labels'], ' | ', out2_old['scores'])
    # print(out2_old['bboxes'])
    # print('*'*200)

def process_folder(folder):
    def get_time(f):
        time_ = f.split('_')[-1]
        time_ = time_[:time_.rfind('.')] 
        return time_ 
    
    files = glob(folder+'/*')
    transition_files = [f for f in files if 'transition' in f ]
    # and 'transition_time_33.214048.jpg' in f
    transition_times = []
    for f in transition_files:
        time = get_time(f)
        transition_times.append(time)
        ref_path = [f for f in files if time in f and 'ref_' in f][0]
        zoomin_paths = [f for f in files if time in f and 'zoomin_' in f ]
        # and 'zoomin_time_reftime_33.214048_36.912907.jpg' in f
        # print('zoomin : ',time,zoomin_paths)
        for zoomin_path in zoomin_paths:
            # print(time,ref_path, zoomin_path,f)
            # t1= TIME.time()
            # inference(ref_path,f,zoomin_path)
            # t2 = TIME.time()
            inference_v2(ref_path,f,zoomin_path)
            # t3 = TIME.time()
            
            # print(t2-t1,t3-t2)
            
            
        # print(time)
    
    #zoomin dont have transition
    ref_paths = [f for f in files if 'ref_' in f and get_time(f) not in transition_times]
    for ref_path in ref_paths:
        zoomin_paths = [f for f in files if 'zoomin_' in f and get_time(ref_path) in f]
        if len(zoomin_paths) > 1:
            transition = min(zoomin_paths,key=get_time)
            # print('replacing : ',transition)
            for zoomin_path in zoomin_paths:
                if transition == zoomin_path:
                    continue
                inference_v2(ref_path,transition,zoomin_path)

folders = [f for f in glob('FE_input/*') if os.path.isdir(os.path.join(f,'images'))]
print(folders)
for folder in folders:
    if 'IMG_1274' not in folder:
        continue
    process_folder(os.path.join(folder,'images'))

# ref_image = cv2.imread('FE_input/IMG_1222_new/images/ref_time_33.214048.jpg')
# transition_image = cv2.imread('FE_input/IMG_1222_new/images/transition_time_33.214048.jpg')
# zoomin_image = cv2.imread('FE_input/IMG_1222_new/images/zoomin_time_reftime_33.214048_38.033385.jpg')

# out_transition, ref_memory, ref_memory_add_pos, transition_memory = model.inference_without_transition(transition_image,ref_image.copy(),0.3)
# out_image = vis(transition_image, out_transition,[])
# cv2.imwrite('transition.jpg',out_image)

# out_zoomin = model.inference_with_transition(zoomin_image.copy(),ref_memory,ref_memory_add_pos,transition_memory,0.3)
# print(out_zoomin)
# out_image = vis(zoomin_image, out_zoomin,[])
# cv2.imwrite('zoomin.jpg',out_image)


# out_zoomin, ref_memory,_, transition_memory = model.inference_without_transition(zoomin_image.copy(),ref_image,0.3)
# # print(out_zoomin)
# out_image = vis(zoomin_image, out_zoomin,[])
# cv2.imwrite('zoomin_old.jpg',out_image)