import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as A
from itertools import combinations

def try_load_json(fp):
    for enc in ['utf-8','gbk','utf-16','latin1']:
        try:
            with open(fp, 'r', encoding=enc) as f:
                return json.load(f)
        except:
            continue
    return None

def visualize_mask(mask, class_mapping):
    h,w = mask.shape
    vis = np.zeros((h,w,3), dtype=np.uint8)
    hid, lid = class_mapping.get('H',-1), class_mapping.get('L',-1)
    vis[mask==hid] = [0,0,255]
    vis[mask==lid] = [0,255,255]
    vis[mask==0]   = [0,0,0]
    return vis

# 基础增强操作
AUG_OPS = {
    'hflip': A.HorizontalFlip(p=1.0),
    'rotate': A.Rotate(limit=30, p=1.0),
    'color': A.RandomBrightnessContrast(p=1.0)
}
def build_augment_pipeline(ops, out_sz):
    tr = [A.Resize(*out_sz, p=1.0)]
    for op in ops:
        tr.append(AUG_OPS[op])
    return A.Compose(tr)

# 生成所有组合
base_augs = ['hflip','rotate','color']
combo_list = [c for i in range(1,len(base_augs)+1) for c in combinations(base_augs,i)]

def convert_labelme_to_segmentation(
    img_dir, json_dir, output_dir,
    train_ratio=0.7, random_seed=42,
    resize_train=True, train_resize_shape=(640,640),
    apply_augment=True
):
    # Normalize to list
    img_dirs  = [img_dir] if isinstance(img_dir,str) else img_dir
    json_dirs = [json_dir] if isinstance(json_dir,str) else json_dir

    # 输出 structure
    for sub in ['train','val']:
        for folder in ['images','masks','vis']:
            os.makedirs(os.path.join(output_dir,sub,folder), exist_ok=True)

    # 类映射及类别计数
    class_mapping = {'_background_':0}
    next_cid = 1
    data_list = []

    # 遍历 JSON，寻找图像
    for jd in json_dirs:
        for js in glob(os.path.join(jd, '*.json')):
            d = try_load_json(js)
            if not d: continue
            ip = d.get('imagePath')
            # 修正标签 key 提及
            img_path = ip if os.path.isabs(ip) else os.path.join(os.path.dirname(js), ip)
            if not os.path.exists(img_path):
                # 在所有 img_dirs 寻找
                cand = None
                for idp in img_dirs:
                    p = os.path.join(idp, os.path.basename(ip))
                    if os.path.exists(p):
                        cand = p; break
                if not cand: continue
                img_path = cand

            img = cv2.imread(img_path)
            if img is None: continue
            h,w = img.shape[:2]
            mask = np.zeros((h,w), np.uint8)

            for shape in d.get('shapes', []):
                lab = shape.get('label','')
                # 修正误标签
                if lab == 'red':
                    lab = 'H'
                elif lab == 'yellow':
                    lab = 'L'
                if lab not in class_mapping:
                    class_mapping[lab] = next_cid; next_cid += 1
                cid = class_mapping[lab]

                pts = shape.get('points',[])
                if len(pts)<3: continue
                pts = [(int(x),int(y)) for x,y in pts]
                mp = Image.new('L',(w,h),0)
                dr = ImageDraw.Draw(mp)
                dr.polygon(pts, outline=cid, fill=cid)
                m_np = np.array(mp, np.uint8)
                mask[m_np==cid] = cid

            name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
            data_list.append((img,mask,name))

    if not data_list:
        print("无可处理样本")
        return

    if train_ratio>=1:
        train_data, val_data = data_list, data_list
    else:
        train_data, val_data = train_test_split(data_list, train_size=train_ratio, random_state=random_seed)

    def process_subset(subdata, subset, augment):
        recs = []
        for img, mask, name in subdata:
            base_img, base_mask = img.copy(), mask.copy()
            # 原始样本
            img2, mask2 = base_img.copy(), base_mask.copy()
            if resize_train and subset=='train':
                img2 = cv2.resize(img2, train_resize_shape, interpolation=cv2.INTER_LINEAR)
                mask2 = cv2.resize(mask2, train_resize_shape, interpolation=cv2.INTER_NEAREST)
            p_img = os.path.join(output_dir, subset, 'images', name)
            p_msk = os.path.join(output_dir, subset, 'masks', name)
            p_vis = os.path.join(output_dir, subset, 'vis', name)
            cv2.imwrite(p_img, img2); cv2.imwrite(p_msk, mask2)
            cv2.imwrite(p_vis, visualize_mask(mask2, class_mapping))
            recs.append((os.path.relpath(p_img,output_dir), os.path.relpath(p_msk, output_dir)))

            # 增强版本
            if augment and subset=='train':
                for ops in combo_list:
                    suffix = '_aug_' + '_'.join(ops)
                    aug = build_augment_pipeline(list(ops), train_resize_shape)
                    tmp = aug(image=base_img, mask=base_mask)
                    imgn, maskn = tmp['image'], tmp['mask']
                    outn = os.path.splitext(name)[0] + suffix + '.png'
                    p_img2 = os.path.join(output_dir, subset, 'images', outn)
                    p_msk2 = os.path.join(output_dir, subset, 'masks', outn)
                    p_vis2 = os.path.join(output_dir, subset, 'vis', outn)
                    cv2.imwrite(p_img2, imgn); cv2.imwrite(p_msk2, maskn)
                    cv2.imwrite(p_vis2, visualize_mask(maskn, class_mapping))
                    recs.append((os.path.relpath(p_img2,output_dir), os.path.relpath(p_msk2, output_dir)))
        return recs

    train_recs = process_subset(train_data, 'train', apply_augment)
    val_recs = process_subset(val_data, 'val', False)

    # 保存 txt
    with open(os.path.join(output_dir,'train.txt'),'w') as f:
        for a,b in train_recs:
            f.write(f"{a.replace(os.sep,'/')} {b.replace(os.sep,'/')}\n")
    with open(os.path.join(output_dir,'val.txt'),'w') as f:
        for a,b in val_recs:
            f.write(f"{a.replace(os.sep,'/')} {b.replace(os.sep,'/')}\n")

    with open(os.path.join(output_dir,'class_mapping.txt'),'w',encoding='utf-8') as f:
        for k,v in class_mapping.items():
            f.write(f"{k}: {v}\n")

    print(f"完成 ✅ train样本数：{len(train_recs)}，val样本数：{len(val_recs)}，类别数：{len(class_mapping)}")

if __name__ == '__main__':
    convert_labelme_to_segmentation(
        # img_dir = [r'C:\Users\Starry\Documents\xwechat_files\wxid_8bivkm7i3b4r22_4a2d\msg\file\2025-06\image\image\pic',
        #            r'C:\Users\Starry\Documents\xwechat_files\wxid_8bivkm7i3b4r22_4a2d\msg\file\2025-06\house2\house2'
        #            ],
        # json_dir = [
        #     r'C:\Users\Starry\Documents\xwechat_files\wxid_8bivkm7i3b4r22_4a2d\msg\file\2025-06\gt\gt',
        #     r'C:\Users\Starry\Documents\xwechat_files\wxid_8bivkm7i3b4r22_4a2d\msg\file\2025-06\house2label\house2label'
        # ],
        img_dir=r'./rawdata/pic',
        json_dir=r'./rawdata/json',
        output_dir = './data/Dataset1',
        train_ratio = 1,
        random_seed = 42,
        resize_train = True,
        train_resize_shape = (640,640),
        apply_augment = False
    )
