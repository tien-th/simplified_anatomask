import copy
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from dataset import MedicalImageReportDataset, load_with_augment, add_gaussian_noise, augment_rotation
from torch.utils.data import DataLoader
from scheduler import LinearWarmupCosineAnnealingLR

# Initialize models, optimizer, and scheduler
def initialize_training(student_model, lr=0.001, warmup_epochs=20, n_epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('+' * 100 )
    print(f"Using {device} device")
    # Create models
    student_model = student_model.to(device)
    teacher_model = copy.deepcopy(student_model)  # Initialize teacher with student weights
    teacher_model = teacher_model.to(device)
    
    # Set up EMA model
    from ema import ModelEMA
    model_ema = ModelEMA(teacher_model)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=n_epochs, warmup_start_lr=1e-6)

    return model_ema, optimizer, scheduler, device


def main():
    # Create dataloader with dummy data
    vision_ssl_paths = ['/home/user01/aiotlab/thaind/PET-CT-2022/split_body']
    image_text_pairs_paths = ['/home/user01/aiotlab/thaind/DAC001_CTAC3.75mm_H_1001_PETWB3DAC001']
    

    batch_size = 8
    n_epoch = 100

    exp_name = f'anatomask_{n_epoch}_epochs_{batch_size}_batch_aug'

    ds = MedicalImageReportDataset(vision_ssl_paths=vision_ssl_paths, image_text_pairs_path=image_text_pairs_paths, split='train', augment=add_gaussian_noise)
    train_data_loader = DataLoader(ds, num_workers=4, batch_size=batch_size, shuffle=True)

    # model 
    from cvit import CTViT
    model = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )
    
    model.load_state_dict(torch.load('/home/user01/aiotlab/htien/pet-clip/ViT_ckpts/CTVit.39000.pt'), strict=False)
    # Initialize models and training components

    model_ema, optimizer, scheduler, device = initialize_training(model, lr=0.001, warmup_epochs=20, n_epochs=n_epoch)

    from log import Logger
    logger = Logger(experiment_name=exp_name)
    
    # Run training
    from train_annatomask import anatomask_training
    trained_model, trained_teacher, losses, ema_losses = anatomask_training(
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_loader=train_data_loader,
        n_epoch=n_epoch,
        gamma = 0.6,
        logger=logger,
        clip=1.0,
        alpha=0.9,
        AMP=False,
        device=device,
        res_dir= f'./results/{exp_name}'
    )

    print("Training completed successfully!")
    return trained_model, trained_teacher

if __name__ == "__main__":
    main()