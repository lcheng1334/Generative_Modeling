"""
VAE Trainer
负责模型训练的主循环
"""
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from  torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

class VAETrainer:
    def __init__(self, 
                 model, 
                 optimizer_config: dict,
                 device: str = 'cuda',
                 log_dir: str = 'runs'):
        
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.get('lr', 0.005),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
        
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=optimizer_config.get('scheduler_gamma', 0.95)
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        
    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx} Train')
        for batch_idx, batch in enumerate(pbar):
            # 获取图像数据（ViewSpecificDataset返回的是dict）
            real_img = batch['image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            results = self.model(real_img)
            
            # Loss computation
            train_loss = self.model.loss_function(
                *results,
                M_N = 0.00025 # kld_weight
            )
            
            train_loss['loss'].backward()
            self.optimizer.step()
            
            total_loss += train_loss['loss'].item()
            
            # Update pbar
            pbar.set_postfix({'Loss': train_loss['loss'].item()})
            
            # Tensorboard logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar(
                    'Loss/train', 
                    train_loss['loss'].item(), 
                    epoch_idx * len(dataloader) + batch_idx
                )
                
        return total_loss / len(dataloader)
        
    def validate(self, dataloader, epoch_idx):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                real_img = batch['image'].to(self.device)
                
                results = self.model(real_img)
                loss = self.model.loss_function(
                    *results,
                    M_N = 0.00025
                )
                total_loss += loss['loss'].item()
                
                # Save reconstruction samples for first batch
                if batch_idx == 0:
                    recons = results[0]
                    # Denormalize for visualization (approximate)
                    vutils.save_image(
                        torch.cat([real_img[:8], recons[:8]], dim=0),
                        self.log_dir / f"recons_{epoch_idx}.png",
                        nrow=8,
                        normalize=True
                    )
                    
        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch_idx)
        return avg_loss
        
    def train(self, train_loader, val_loader, epochs=10, save_path='best_model.pth'):
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
                
        self.writer.close()
