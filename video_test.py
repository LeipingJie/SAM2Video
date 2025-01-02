import os
import argparse
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--npts', type=int, required=True)
args = parser.parse_args()

model_type = args.type
assert model_type in ['tiny', 'small', 'large']
if model_type == 'tiny':
    sam2_checkpoint = "./sam2_hiera_tiny.pt"
    model_cfg = "./sam2_hiera_t.yaml"
elif model_type == 'small':
    sam2_checkpoint = "./sam2_hiera_small.pt"
    model_cfg = "./sam2_hiera_s.yaml"
elif model_type == 'large':
    sam2_checkpoint = "./sam2_hiera_large.pt"
    model_cfg = "./sam2_hiera_l.yaml"

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


# Loading the SAM 2 video predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
predictor = predictor.to('cuda:0')

dataname = args.dataset
image_folder_name = ''
label_folder_name = ''
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# video shadow
# visha
# video mirror
# vmd
# salient object
# mose, lvosv2, USOD10k(underwater)

supported_video_datasets = ['visha', 'vmd']
assert dataname in supported_video_datasets, f'invalid dataset {dataname}, choose from {supported_video_datasets}'
if dataname == 'visha':
    data_root = f'./dataset/shadow/visha/'
    image_folder_name, label_folder_name = 'images', 'labels'
elif dataname == 'vmd':
    data_root = f'./dataset/mirror/VMD/'
    image_folder_name, label_folder_name = 'JPEGImages', 'SegmentationClassPNG'

# splits = ['train', 'test']
splits = ['test']
prompt_type = args.prompt # mask / point
gen_first_frame_mask_only = True
gen_images = True
n_pos_pts = args.npts
n_neg_pts = args.npts
prefix_dir = f'2output'
ours_dir_root = f'./{prefix_dir}_{dataname}_{prompt_type}_{model_type}_{n_neg_pts}'

if gen_images:
    os.makedirs(ours_dir_root, exist_ok=True)
    for split in splits:
        # create save folder
        ours_dir_split = os.path.join(ours_dir_root, split)
        os.makedirs(ours_dir_split, exist_ok=True)
        
        if dataname == 'vmd':
            split_image_root = os.path.join(data_root, split)
            split_label_root = os.path.join(data_root, split)
            dirs = os.listdir(os.path.join(data_root, split))
        elif dataname == 'offical':
            dirs = os.listdir(os.path.join(data_root, split))
        elif dataname == 'vcod':
            dirs = os.listdir(data_root)
            split_image_root = data_root
            split_label_root = data_root
        else:
            split_image_root = os.path.join(data_root, split, image_folder_name)
            split_label_root = os.path.join(data_root, split, label_folder_name)
            dirs = os.listdir(split_image_root)

        print(f"processing {split}, number of videos: {len(dirs)}")
        for idx, d in enumerate(dirs):
            if not os.path.isdir(os.path.join(split_image_root, d)): continue
            # create save folder
            ours_dir = os.path.join(ours_dir_split, d)
            os.makedirs(ours_dir, exist_ok=True)

            # if d!='white_car':
            folder_files = os.listdir(ours_dir)
            video_image_dir = os.path.join(split_image_root, d)
            video_label_dir = os.path.join(split_label_root, d)
            if dataname == 'vmd':
                video_image_dir = os.path.join(video_image_dir, image_folder_name)
                video_label_dir = os.path.join(video_label_dir, label_folder_name)
            elif dataname == 'vcod':
                video_image_dir = os.path.join(video_image_dir, image_folder_name)
                video_label_dir = os.path.join(video_label_dir, label_folder_name)

            frame_names = [fn for fn in os.listdir(video_image_dir) if fn.endswith(".jpg") or fn.endswith(".png")]
            frame_names.sort()
            print(f"[{idx:03d}/ {len(dirs)}]>>> {d}, len: {len(frame_names)}")
            if (len(frame_names) <=0 or len(frame_names)==len(folder_files)):
                continue
            
            inference_state = predictor.init_state(video_path=video_image_dir)
            predictor.reset_state(inference_state)

            # set prompt for the first frame
            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            frame_image_path = os.path.join(video_image_dir, frame_names[ann_frame_idx])
            frame_label_path = os.path.join(video_label_dir, frame_names[ann_frame_idx].split('.')[0]+".png")
            video_img = np.array(Image.open(frame_image_path))
            video_mask = np.array(Image.open(frame_label_path))
            # print(frame_names[ann_frame_idx])
            
            if prompt_type=='point':
                h, w, c = video_img.shape
                coordPos = np.argwhere(video_mask>0)
                coordNeg = np.argwhere(video_mask==0)
                pos_pt_idx = np.random.randint(0, len(coordPos), (n_pos_pts))
                neg_pt_idx = np.random.randint(0, len(coordNeg), (n_neg_pts))

                if gen_first_frame_mask_only:
                    plt.figure(figsize=(12, 8))
                    plt.title(f"frame {ann_frame_idx}")
                    plt.imshow(video_img)

                labels = np.array([1]* n_pos_pts + [0]*n_neg_pts, np.int32)
                points = np.zeros((n_pos_pts+n_neg_pts, 2), dtype=np.float32)
                for idx, idx_pt in enumerate(pos_pt_idx):
                    coordY, coordX = coordPos[idx_pt]
                    points[idx] = [coordX, coordY]

                for idx, idx_pt in enumerate(neg_pt_idx):
                    coordY, coordX = coordNeg[idx_pt]
                    points[idx+n_pos_pts] = [coordX, coordY]

                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
                if gen_first_frame_mask_only: 
                    show_points(points, labels, plt.gca())
                    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                if gen_first_frame_mask_only: 
                    # show_points(points, labels, plt.gca())
                    # plt.imshow(Image.open(frame_image_path))
                    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
                    tmp_name = os.path.basename(frame_image_path)
                    pred_save_path = os.path.join(ours_dir, tmp_name)
                    plt.savefig(pred_save_path, bbox_inches='tight')
                    plt.close()
                    # plt.savefig("test.jpg")
                    continue
                    # assert 1==0
            elif prompt_type=='mask':
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        mask=video_mask
                    )
            elif prompt_type == 'bbox':
                pass
                
            # get other predictions
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                pred_mask = ((out_mask_logits > 0.0).cpu().squeeze().numpy()*255).astype(np.uint8)
                pred_save_path = os.path.join(ours_dir, frame_names[out_frame_idx][:-3]+'png')
                # print(">>> ", frame_names[out_frame_idx], pred_save_path)
                Image.fromarray(pred_mask).save(pred_save_path)