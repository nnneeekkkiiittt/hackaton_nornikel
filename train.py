import os
from dataset import OreDataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import torch

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(imgs)
            loss = criterion(preds, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    tp, fp, fn = 0, 0, 0
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, masks)
        total_loss += loss.item()
        
        preds_class = preds.argmax(dim=1)
        tp += ((preds_class == 1) & (masks == 1)).sum().item()
        fp += ((preds_class == 1) & (masks == 0)).sum().item()
        fn += ((preds_class == 0) & (masks == 1)).sum().item()
        
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    return total_loss / len(loader), f1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    img_dir = 'dataset/ores_by_grade/patches/images'
    mask_dir = 'dataset/ores_by_grade/patches/masks'

    filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    np.random.seed(42)
    np.random.shuffle(filenames)
    split_idx = int(0.8 * len(filenames))
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    train_tf = A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    val_tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_ds = OreDataset(img_dir, mask_dir, train_files, train_tf)
    val_ds = OreDataset(img_dir, mask_dir, val_files, val_tf)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode='multiclass', classes=2)
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    
    criterion = lambda y_pred, y_true: dice_loss(y_pred, y_true) + focal_loss(y_pred, y_true)
    scaler = torch.amp.GradScaler('cuda')
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    best_f1 = 0.0
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    for epoch in range(5):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        print(f"Frozen Epoch {epoch+1}/5 | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    for epoch in range(25):
        loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"FT Epoch {epoch+6}/30 | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_unet_talc.pth")
            print(f"Saved best checkpoint with F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()