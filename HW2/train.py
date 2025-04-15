import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import v2

from torchvision import transforms
import json
import pickle


class COCODetection(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_dir, path)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            img, boxes, labels = self.transforms(img, boxes, labels)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }


        assert img.shape[0] == 3, f"Image has {img.shape[0]} channels instead of 3"

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))

class DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def get_model(num_classes, mode = 0):
    # anchor_generator = AnchorGenerator(
    #     sizes=((16,), (32,), (64,), (128,), (256,), (512,)),
    #     aspect_ratios=((0.25, 0.5, 1.0, 2.0),) * 5
    # )

    # model = fasterrcnn_resnet50_fpn_v2(weights='FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1', anchor =  anchor_generator)

    model = fasterrcnn_resnet50_fpn_v2(weights='FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1')
    
    # model.rpn.anchor_generator = anchor_generator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if mode == 0:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif mode == 1:
        model.roi_heads.box_predictor = DropoutFastRCNNPredictor(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.backbone.named_parameters():
        if name.startswith("fpn"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for name, param in model.backbone.body.named_parameters():
        if name.startswith("layer4"):
            param.requires_grad = True

    # # Unfreeze RPN and RoI heads
    for param in model.rpn.parameters():
        param.requires_grad = True
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    # for name, param in model.roi_heads.named_parameters():
    #     if "box_predictor" in name or "box_head" in name:
    #         param.requires_grad = True

    return model


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return int(os.environ["RANK"]) == 0

@torch.no_grad()
def evaluate_coco(model, data_loader, coco_gt, device, history = None):
    model.eval()
    local_results = []
    total_labels = torch.tensor(0, dtype=torch.int32, device=device)
    correct_labels = torch.tensor(0, dtype=torch.int32, device=device)

    for images, targets in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()
            image_id = int(target['image_id'].item())

            gt_labels = target['labels'].cpu()
            gt_boxes = target['boxes'].cpu()
            total_labels += len(gt_labels)

            matched = min(len(gt_labels), len(labels))
            correct_labels += (labels[:matched][torch.argsort(boxes[:matched].T, dim=1)[0]] == gt_labels[:matched][torch.argsort(gt_boxes[:matched].T, dim=1)[0]]).sum().item()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                local_results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score)
                })

    # Gather predictions from all processes
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, local_results)

    dist.all_reduce(correct_labels, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_labels, op=dist.ReduceOp.SUM)

    # Flatten list of lists
    if is_main_process():
        all_results = [item for sublist in gathered_results for item in sublist]
        with open('val_res.json', "w") as f:
            json.dump(all_results, f, indent=2)
        coco_dt = coco_gt.loadRes(all_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        
        if total_labels > 0:
            label_acc = correct_labels / total_labels
            print(f"✅ Label Accuracy: {label_acc:.4f} ({correct_labels}/{total_labels})")
        
        
        if history != None:
            map_50_95 = coco_eval.stats[0]
            history.append({'evalMAP' : map_50_95, 'label_acc' : label_acc.cpu()})

def train(load_checkpoint = False, model_path="fasterrcnn_ddp_4.pth", mode = 0):
    setup_ddp()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Hyperparameters
    num_classes = 11
    num_epochs = 10
    batch_size = 5

    # Dataset and transforms
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    # transform = v2.Compose([
    #     v2.ToImage(), 
    #     v2.RandomHorizontalFlip(0.5),
    #     v2.RandomRotation(20),
    #     v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     v2.ToDtype(torch.float32, scale=True)
    # ])

    transform_v = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_dataset = COCODetection('data/train', 'data/train.json', transforms=transform)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_dataset = COCODetection('data/valid', 'data/valid.json', transforms=transform_v)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_coco = val_loader.dataset.coco

    # Model setup
    model = get_model(num_classes, mode).to(device)
    print(model.transform)
    # torch.load path if needed
    if load_checkpoint:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer and AMP scaler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scaler = GradScaler()

    training_stats = []
    val_stats = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", unit="batch") if is_main_process() else train_loader
        epoch_loss = 0.0

        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                training_stats.append(loss_dict)
                # loss = loss_dict['loss_classifier']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if is_main_process():
                pbar.set_postfix(loss=loss.item())

        if is_main_process():
            avg_loss = epoch_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            torch.save(model.module.state_dict(), f'fasterrcnn_ddp_{epoch}.pth')
            print("✅ Model saved as fasterrcnn_ddp_{epoch}.pth")

        evaluate_coco(model.module, val_loader, val_coco, device, val_stats)

    
    if is_main_process():
        with open("training_stats.pkl", "wb") as f:
            pickle.dump(training_stats, f)

        with open("val_stats.pkl", "wb") as f:
            pickle.dump(val_stats, f)

    cleanup_ddp()


def load_and_evaluate(model_path="fasterrcnn_ddp_4.pth", mode = 0):
    setup_ddp()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    num_classes = 11
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    val_dataset = COCODetection('data/valid', 'data/valid.json', transforms=transform)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                            num_workers=4, collate_fn=collate_fn)
    val_coco = val_loader.dataset.coco

    model = get_model(num_classes, mode)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 🔥 load directly
    model.to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # if is_main_process():
    print("📦 Loaded model weights. Running evaluation...")
    evaluate_coco(model.module, val_loader, val_coco, device)

    cleanup_ddp()


class TestImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.transform = transform if transform else transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_files[idx]  # Return file name as ID

    def __len__(self):
        return len(self.image_files)

@torch.no_grad()
def test_on_folder(model_path="fasterrcnn_ddp_4.pth", mode = 0, device="cuda:1"):

    image_dir = 'data/test'
    output_file="submission.json"
    

    num_classes = 11
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])
    model = get_model(num_classes, mode)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 🔥 load directly
    model.to(device)

    model.eval()
    dataset = TestImageFolderDataset(image_dir, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    results = []

    for images, filenames in tqdm(loader, desc="Testing"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, fname in zip(outputs, filenames):
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()
            labels = output['labels'].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                width, height = x2 - x1, y2 - y1
                results.append({
                    "image_id": int(fname[:-4]),
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "score": float(score)
                })

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default='train')
    parser.add_argument('--chkt-flag', default=False)
    parser.add_argument('--model-path', default='fasterrcnn_ddp.pth')
    parser.add_argument('--model-type', choices=[0, 1], default=0, type= int)
    parser.add_argument('--local-rank', default='0')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.chkt_flag,args.model_path,args.model_type)
    elif args.mode == 'eval':
        load_and_evaluate(args.model_path, args.model_type)
    elif args.mode == 'test':
        test_on_folder(args.model_path, args.model_type)

