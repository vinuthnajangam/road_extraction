import torch
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm
from model import UNetWithPretrainedEncoder
from dataset import RoadExtractionDataset
import segmentation_models_pytorch as smp

def train_fn(model, loader, opt, bce, dice, scaler):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    for _, (sat, mask) in enumerate(loop):
        sat = sat.to(config.DEVICE, dtype=torch.float32)
        mask = mask.to(config.DEVICE, dtype=torch.float32)
        mask = mask.unsqueeze(1)

        with torch.amp.autocast(device_type=config.DEVICE):
            pred = model(sat)
            loss = bce(pred, mask) + dice(pred, mask)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())
        
def val_fn(model, loader, bce, dice):
    model.eval()
    loop = tqdm(loader, leave=True, desc="Validation")
    
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for _, (sat, mask) in enumerate(loop):
            sat = sat.to(config.DEVICE, dtype=torch.float32)
            mask = mask.to(config.DEVICE, dtype=torch.float32)
            mask = mask.unsqueeze(1)

            pred = model(sat)
            loss = bce(pred, mask) + dice(pred, mask)
            total_loss += loss.item()
            
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            total_iou += iou.item()
            
            loop.set_postfix(loss=loss.item(), iou=iou.item())
            
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_iou

def main():
    model = UNetWithPretrainedEncoder(encoder_name='resnet34', in_channels=3, num_classes=1).to(config.DEVICE)

    opt = optim.Adam(
        list(model.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode='binary')
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_gen_B,
            model,
            opt,
            config.LEARNING_RATE,
        )

    dataset = RoadExtractionDataset(
        root_sat=os.path.join(config.TRAIN_DIR, "satellite"),
        root_mask=os.path.join(config.TRAIN_DIR, "masks"),
        transform=config.train_transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_dataset = RoadExtractionDataset(
        root_sat=os.path.join(config.VAL_DIR, "satellite"),
        root_mask=os.path.join(config.VAL_DIR, "masks"),
        transform=config.val_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    scaler = torch.amp.GradScaler()
    
    best_val_iou = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        train_fn(
            model,
            loader,
            opt,
            bce,
            dice,scaler
        )
        
        val_loss, val_iou = val_fn(
            model,
            val_loader,
            bce,
            dice
        )
        
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if config.SAVE_MODEL and val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(model, opt, filename=config.CHECKPOINT)
            print(f"Model saved with IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()