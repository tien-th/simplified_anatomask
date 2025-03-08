import torch
from torch.cuda.amp import GradScaler
import math
import sys

from tqdm import tqdm

import numpy as np
import os 

@torch.no_grad()
def generate_mask(loss_pred, gamma, epoch, total_epoch, guide = True):
    B, c, d, h, w = loss_pred.shape
    L = c * d * h * w
    loss_pred = loss_pred.view(B, L)

    len_keep = int(L * (1 - gamma))
 
    ids_shuffle_loss = torch.argsort(loss_pred, dim=1)  # (N, L)
    keep_ratio = 2/3
    ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()
    len_loss = 0

    if guide:
        ### easy to hard
        keep_ratio = float((epoch + 1) / total_epoch) * 0.5

        ### hard-to-easy
        # keep_ratio = 0.5 - float(epoch / total_epoch) * 0.5

        ## top 0 -> 0.5
    
    for i in range(B):
        ## mask top `keep_ratio` loss and `1 - keep_ratio` random
        len_loss = int((L - len_keep) * keep_ratio)
        easy_len = int((L - len_keep)) - len_loss

        ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]
        temp = torch.arange(L, device=loss_pred.device)
        deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
        np.random.shuffle(deleted)
        ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # generate mask: 1 is keep, 0 is remove
    mask = torch.zeros([B, L], device=loss_pred.device, dtype=torch.bool)  # Changed from ones to zeros
    mask[:, : len_keep] = 1  # Changed from 0 to 1
    # unshuffle to get final mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.view(B, 1, h, w, d)


def generate_random_mask(input_size, mask_ratio, device='cuda'):
    """
    Generates a random binary mask for batched voxel data.

    Args:
        input_size (tuple): Shape of the input tensor (B, C, D, H, W).
        mask_ratio (float): Mask ratio in each data voxel (0.0 to 1.0).
        device (str, optional): Device to create the mask on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        torch.Tensor: Binary mask of the same shape as input_size.
                       1 indicates masked (kept), 0 indicates masked out.
    """
    assert len(input_size) == 5, "Input size should be (B, C, D, H, W) for batched voxels."
    B, C, D, H, W = input_size

    mask = torch.zeros(input_size, device=device) # Initialize mask with zeros

    for b in range(B):
        # Calculate total elements in a single voxel (C, D, H, W)
        voxel_size = (C, D, H, W)
        total_elements_voxel = torch.tensor(voxel_size).prod().item()
        num_ones_voxel = int(mask_ratio * total_elements_voxel)

        # Create a flat mask for the voxel
        mask_flat_voxel = torch.ones(total_elements_voxel, device=device)
        indices_voxel = torch.randperm(total_elements_voxel, device=device)[:num_ones_voxel]
        mask_flat_voxel[indices_voxel] = 0.0

        # Reshape the flat voxel mask back to (C, D, H, W) and assign to the batch
        mask[b, :, :, :, :] = mask_flat_voxel.view(voxel_size)

    return mask


def input_masked(input_tensor, mask):
    return input_tensor * mask


def forward_teacher_network(teacher_model, masked_input, original_input):
    with torch.no_grad():
        reconstruction = teacher_model(masked_input)
        L_rec = (reconstruction - original_input) ** 2  # Mean squared error per voxel
    return L_rec


def print_to_log_file(*args):
    print(*args)

def forward_loss(inp, rec, active_b1ff):
    mean = inp.mean(dim=-1, keepdim=True)
    var = inp.var(dim=-1, keepdim=True)
    inp = (inp - mean) / (var + 1.e-6) ** .5  #

    l2_loss = ((rec - inp) ** 2)  # (B, L, C) ==mean==> (B, L)
    non_active = 1 - active_b1ff 
    l2_loss = l2_loss
    recon_loss = l2_loss.mul_(non_active).sum() / (
            non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

    return recon_loss

# Modified AnatomaSK training function with elements from both code snippets
def anatomask_training(
    model,
    model_ema,
    optimizer,
    scheduler,
    train_data_loader,
    n_epoch,
    gamma, # Masking ratio
    logger,
    clip=1.0,
    alpha=0.9,
    AMP=True,
    device='cuda',
    res_dir = './results'
):
    """
    Main training function implementing the anatomask pipeline with dynamic masking ratio
    and EMA model updates.
    """

    os.makedirs(res_dir, exist_ok=True)

    epoch_loss = []
    epoch_ema_loss = []

    scaler = GradScaler() if AMP else None

    for i in tqdm(range(n_epoch)):
        model.train()
        per_loss = 0.0
        per_p_loss = 0.0

        print_to_log_file('')
        print_to_log_file(f'Epoch {i}')
        print_to_log_file()

        # print_to_log_file(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
        logger.log('epoch', i)
        logger.log('learning_rate', optimizer.param_groups[0]['lr'])
        epoch_start_timestamps = time.time()
        if i < n_epoch // 4:
            model_ema.decay = 0.999 + i / (n_epoch // 4) * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

        logger.log('ema_decay', model_ema.decay)


        for batch in tqdm(train_data_loader):
            input_data = batch.to(device)

            if AMP:
                with torch.cuda.amp.autocast():
                    # Generate random initial mask
                    M_init = generate_random_mask(input_data.shape, gamma, device)
                    masked_input = input_masked(input_data, M_init)

                    # Forward through teacher (EMA) model to get reconstruction loss
                    with torch.no_grad():
                        L_rec = forward_teacher_network(model_ema, masked_input, input_data)
                    M_final = generate_mask(L_rec, gamma, epoch = i, total_epoch=n_epoch)
                    # Mask input with final mask and train student network
                    student_masked_input = input_masked(input_data, M_final)

                    # Forward student model
                    student_reconstruction = model(student_masked_input)
                    loss = forward_loss(input_data, student_reconstruction, M_final)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()

            else:
                # Generate random initial mask
                M_init = generate_random_mask(input_data.shape, gamma, device)
                masked_input = input_masked(input_data, M_init)

                # Forward through teacher (EMA) model to get reconstruction loss
                with torch.no_grad():
                    L_rec = forward_teacher_network(model_ema, masked_input, input_data)

                # Generate final mask based on reconstruction loss
                M_final = generate_mask(L_rec, gamma, epoch = i, total_epoch=n_epoch)
                # Mask input with final mask and train student network
                student_masked_input = input_masked(input_data, M_final)

                # Forward student model
                student_reconstruction = model(student_masked_input)
                loss = forward_loss(input_data, student_reconstruction, M_final)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            # Update EMA model
            model_ema.update(model)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(loss_value)
                print(f'Loss is {loss_value}, stopping training!')
                sys.exit(-1)

            per_loss += loss_value
            per_p_loss += loss_value

            torch.cuda.synchronize()

        scheduler.step()
        epoch_end_timestamps = time.time()

        time = epoch_end_timestamps - epoch_start_timestamps
        print_to_log_file(f'Epoch {i} took {time} seconds')

        avg_loss = per_loss / len(train_data_loader)
        epoch_loss.append(avg_loss)

        # Calculate EMA loss
        if i == 0:
            ema_loss = alpha * avg_loss + (1 - alpha) * avg_loss
        else:
            ema_loss = alpha * epoch_ema_loss[-1] + (1 - alpha) * avg_loss

        epoch_ema_loss.append(ema_loss)

        print('Epoch ', 'Train AVG Loss: ', avg_loss)
        logger.log('train_losses', avg_loss)
        print('Epoch ', 'Train EMA Loss: ', ema_loss)
        logger.log('train_ema_losses', ema_loss)
        print('Train Pixel Loss: ', per_p_loss / len(train_data_loader))
        logger.log('train_pixel_losses', per_p_loss / len(train_data_loader))

        print_to_log_file('train_loss', np.round(avg_loss, decimals=4))
        
        # Save checkpoint
        checkpoint = {
            'network_weights': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'grad_scaler_state': scaler.state_dict() if scaler else None,
            'train_loss': epoch_loss,
            'scheduler_state': scheduler.state_dict(),
            'current_epoch': i
        }
        torch.save(checkpoint, f'{res_dir}/model_checkpoint_epoch_{i}.pt')

    return model, model_ema, epoch_loss, epoch_ema_loss


