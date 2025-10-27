import os
import time
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import itertools

import pyiqa
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm

import cv2

operation_seed_counter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

class TrainingConfig:
    # --- Data and Augmentation ---
    val_split_ratio = 0.15
    
    # --- Model and Training Hyperparameters ---
    n_feature = 48
    n_channel = 1  # Set to 1 for grayscale X-ray images
    lr = 3e-4
    gamma = 0.5
    # With a small dataset (34 images), more epochs might be beneficial.
    n_epoch = 100 
    # Adjust batch size based on your GPU memory. 4 is a safe start.
    batchsize = 4
    patchsize = 512 # images are already preprocessed to this size
    
    # Loss function weight for the regularization term
    increase_ratio = 1.0

    # --- Paths ---
    save_model_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Nei2Nei/Models'          # Base path to save model checkpoints
    save_losses_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Nei2Nei/Losses'
    save_results_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Nei2Nei/Results'

    def __init__(self):
        self.update_paths()
    
    def get_hyperparameter_string(self):
        return (f"ep{self.n_epoch}_lr{self.lr}_nf{self.n_feature}_"
                f"b{self.batchsize}_p{self.patchsize}_ir{self.increase_ratio}")

    def update_paths(self):
        if self.patchsize == 512:
            self.img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Train_Val_512x512'
            self.test_img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Test_512x512'
        elif self.patchsize == 1024:
            self.img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Train_Val_1024x1024'
            self.test_img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Test_1024x1024'
        elif self.patchsize == 2048:
            self.img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Train_Val_2048x2048'
            self.test_img_dir = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/Test_2048x2048'
        else:
            # Handle unknown patchsize
            raise ValueError(f"Invalid patchsize: {self.patchsize}. No paths set.")

# --- Define the Dataset Class ---
class DicomTensorDataset(Dataset):
    """
    Custom Dataset for loading pre-processed X-ray images.
    Applies specified transforms to each image upon loading.
    """
    def __init__(self, file_paths, transform=None):
        super(DicomTensorDataset, self).__init__()
        self.image_files = file_paths
        self.transform = transform
        # print(f'Initialized dataset with {len(self.image_files)} images.')

    def __getitem__(self, index):
        img_path = self.image_files[index]
        im = Image.open(img_path).convert('L') # Convert to grayscale
        if self.transform:
            im = self.transform(im)
        return im, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_files)

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)

def generate_mask_pair(img):
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n*h//2*w//2*4,), dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(size=(n*h//2*w//2*4,), dtype=torch.bool, device=img.device)
    idx_pair = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.int64, device=img.device)
    rd_idx = torch.randint(0, 8, (n*h//2*w//2,), generator=get_generator(), device=img.device)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0, end=n*h//2*w//2*4, step=4, dtype=torch.int64, device=img.device).reshape(-1, 1)
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h//2, w//2, dtype=img.dtype, layout=img.layout, device=img.device)
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i+1, :, :] = img_per_channel[mask].reshape(n,h//2,w//2,1).permute(0,3,1,2)
    return subimage

class UNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_feature=48):
        super(UNet, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(in_nc, n_feature, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.conv1 = nn.Sequential(nn.Conv2d(n_feature, n_feature, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(nn.Conv2d(n_feature, n_feature*2, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(n_feature*2, n_feature*4, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(nn.Conv2d(n_feature*4, n_feature*8, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bridge = nn.Sequential(nn.Conv2d(n_feature*8, n_feature*16, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                    nn.Conv2d(n_feature*16, n_feature*16, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.up4 = nn.ConvTranspose2d(n_feature*16, n_feature*8, 2, 2)
        self.dconv4 = nn.Sequential(nn.Conv2d(n_feature*16, n_feature*8, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.up3 = nn.ConvTranspose2d(n_feature*8, n_feature*4, 2, 2)
        self.dconv3 = nn.Sequential(nn.Conv2d(n_feature*8, n_feature*4, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.up2 = nn.ConvTranspose2d(n_feature*4, n_feature*2, 2, 2)
        self.dconv2 = nn.Sequential(nn.Conv2d(n_feature*4, n_feature*2, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.up1 = nn.ConvTranspose2d(n_feature*2, n_feature, 2, 2)
        self.dconv1 = nn.Sequential(nn.Conv2d(n_feature*2, n_feature, 3, 1, 1), nn.LeakyReLU(0.1, True))
        self.out_conv = nn.Conv2d(n_feature, out_nc, 3, 1, 1)

    def forward(self, x):
        c1 = self.conv1(self.in_conv(x))
        p1 = self.pool1(c1); c2 = self.conv2(p1)
        p2 = self.pool2(c2); c3 = self.conv3(p2)
        p3 = self.pool3(c3); c4 = self.conv4(p3)
        p4 = self.pool4(c4); b = self.bridge(p4)
        u4 = self.up4(b); merge4 = torch.cat([u4, c4], 1); d4 = self.dconv4(merge4)
        u3 = self.up3(d4); merge3 = torch.cat([u3, c3], 1); d3 = self.dconv3(merge3)
        u2 = self.up2(d3); merge2 = torch.cat([u2, c2], 1); d2 = self.dconv2(merge2)
        u1 = self.up1(d2); merge1 = torch.cat([u1, c1], 1); d1 = self.dconv1(merge1)
        return self.out_conv(d1)

def main(opt):
    """
    This function contains all the logic from your run_experiment function above.
    It takes the 'opt' object as an argument.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define the simplified transformation ---
    online_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # --- Load Files from the New Offline Directories ---
    # Get all file paths from your new augmented train/val folder
    train_val_files = sorted([os.path.join(opt.img_dir, f) for f in os.listdir(opt.img_dir)])
    # Get all file paths from your new un-augmented test folder
    test_files = sorted([os.path.join(opt.test_img_dir, f) for f in os.listdir(opt.test_img_dir)])
    
    print(f"Found {len(train_val_files)} total augmented train/val images.")
    print(f"Found {len(test_files)} total un-augmented test images.")
    
    # Now, split the augmented train_val set into a final training and validation set.
    val_size = int(len(train_val_files) * opt.val_split_ratio)
    train_size = len(train_val_files) - val_size
    generator = torch.Generator().manual_seed(42) # for reproducible splits
    train_files, val_files = random_split(train_val_files, [train_size, val_size], generator=generator)
    train_files, val_files = list(train_files), list(val_files) # Convert to lists
    
    # --- Create Datasets with the simplified transform ---
    train_dataset = DicomTensorDataset(train_files, transform=online_transforms)
    val_dataset = DicomTensorDataset(val_files, transform=online_transforms)
    test_dataset = DicomTensorDataset(test_files, transform=online_transforms)
    
    print(f"\nDataset sizes for this run:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=opt.batchsize, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=0, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)

    network = UNet(in_nc=opt.n_channel, out_nc=opt.n_channel, n_feature=opt.n_feature).to(device)

    optimizer = optim.Adam(network.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x * opt.n_epoch) for x in [0.2, 0.4, 0.6, 0.8]], gamma=opt.gamma)
    
    # --- MODIFIED: Get hyperparam string for filenames ---
    hyperparam_str = opt.get_hyperparameter_string()
    print(f"Using hyperparameter string: {hyperparam_str}")
    
    # --- Model Saving Setup ---
    save_model_dir = os.path.join(opt.save_model_path, hyperparam_str)
    os.makedirs(save_model_dir, exist_ok=True)
    print(f"Models and logs will be saved to: {save_model_dir}")
    
    # --- Lists to store loss history for plotting ---
    train_loss_history = []
    val_loss_history = []
    
    # --- TRAINING & VALIDATION ---
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, opt.n_epoch + 1):
        # --- Training Phase ---
        network.train()
        epoch_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch}/{opt.n_epoch}")
        
        for i, (noisy_image_batch, _) in enumerate(train_iterator):
            noisy = noisy_image_batch.to(device)
    
            optimizer.zero_grad()
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)
    
            with torch.no_grad():
                noisy_denoised = network(noisy)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)
    
            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
    
            Lambda = epoch / opt.n_epoch * opt.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
    
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = loss1 + loss2
    
            loss_all.backward()
            optimizer.step()
            epoch_train_loss += loss_all.item()
            train_iterator.set_postfix({'loss': loss_all.item()})
    
        # --- Validation Phase ---
        network.eval()
        epoch_val_loss = 0
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch}/{opt.n_epoch}")
        with torch.no_grad():
            for i, (noisy_val_image, _) in enumerate (val_iterator):
                noisy_val = noisy_val_image.to(device)
    
                mask1_val, mask2_val = generate_mask_pair(noisy_val)
                noisy_sub1_val = generate_subimages(noisy_val, mask1_val)
                noisy_sub2_val = generate_subimages(noisy_val, mask2_val)
    
                noisy_denoised_val = network(noisy_val)
                noisy_sub1_denoised_val = generate_subimages(noisy_denoised_val, mask1_val)
                noisy_sub2_denoised_val = generate_subimages(noisy_denoised_val, mask2_val)
    
                noisy_output_val = network(noisy_sub1_val)
    
                diff_val = noisy_output_val - noisy_sub2_val
                exp_diff_val = noisy_sub1_denoised_val - noisy_sub2_denoised_val
    
                val_loss1 = torch.mean(diff_val**2)
                val_loss2 = Lambda * torch.mean((diff_val - exp_diff_val)**2)
                val_loss_all = val_loss1 + val_loss2
                epoch_val_loss += val_loss_all.item()
                val_iterator.set_postfix({'val_loss': val_loss_all.item()})
    
        scheduler.step()
    
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
    
        # Append the calculated losses to the lists for plotting
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
    
        print(f"Epoch [{epoch}/{opt.n_epoch}] | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")
    
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_name = f"best_model.pth"
            best_model_path = os.path.join(save_model_dir, best_model_name)
            torch.save(network.state_dict(), best_model_path)
            print(f'   -> New best model saved to {best_model_path}')
    
    # --- Save Final Model ---
    final_model_name = f"final_model.pth"
    final_model_path = os.path.join(save_model_dir, final_model_name)
    torch.save(network.state_dict(), final_model_path)
    print(f'   -> Final model saved to {final_model_path}')
    
    end_time = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    
    # --- MODIFIED: Save Loss History to CSV ---
    print(f"\nSaving loss history to CSV...")
    losses_df = pd.DataFrame({
        'epoch': range(1, opt.n_epoch + 1),
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    })
    losses_csv_name = f"{hyperparam_str}.csv"
    losses_csv_path = os.path.join(opt.save_losses_path, losses_csv_name)
    losses_df.to_csv(losses_csv_path, index=False)
    print(f"Loss history saved to {losses_csv_path}")

    # --- Setup ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    # --- Create IQA metric models ---
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    piqe_metric = pyiqa.create_metric('piqe', device=device)
    brisque_metric = pyiqa.create_metric('brisque', device=device)
    
    # --- Load the Best Model ---
    model_path = os.path.join(save_model_dir, "best_model.pth")
    
    state_dict = torch.load(model_path, weights_only=True)
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Loaded best model for evaluation: {model_path}\n")
    
    # --- Prepare storage ---
    results_data = []
    original_images_np, denoised_images_np, residuals = [], [], []
    psnr_scores = []
    original_piqe_scores, denoised_piqe_scores = [], []
    original_brisque_scores, denoised_brisque_scores = [], []
    original_niqe_scores, denoised_niqe_scores = [], []
    
    test_iterator = tqdm(test_loader, desc="Evaluating Test Set")
    
    # --- Evaluation ---
    with torch.no_grad():
        for i, (original_image_batch, filename_batch) in enumerate(test_iterator):
            original_image_tensor = original_image_batch.to(device)  # (1, 1, H, W)
            filename = filename_batch[0]
    
            # --- Model inference ---
            denoised_image_tensor = network(original_image_tensor).clamp(0, 1)
    
            # --- Compute IQA metrics (directly on tensors, no numpy) ---
            original_niqe = niqe_metric(original_image_tensor).item()
            denoised_niqe = niqe_metric(denoised_image_tensor).item()
            original_piqe = piqe_metric(original_image_tensor).item()
            denoised_piqe = piqe_metric(denoised_image_tensor).item()
            original_brisque = brisque_metric(original_image_tensor).item()
            denoised_brisque = brisque_metric(denoised_image_tensor).item()
    
            # --- Convert to NumPy for PSNR and saving ---
            original_image_np = original_image_tensor.cpu().numpy()[0, 0]
            denoised_image_np = denoised_image_tensor.cpu().numpy()[0, 0]
            residual_np = original_image_np - denoised_image_np
    
            # --- Metrics ---
            psnr = compare_psnr(original_image_np, denoised_image_np)
    
            # --- Store all results ---
            original_images_np.append(original_image_np)
            denoised_images_np.append(denoised_image_np)
            residuals.append(residual_np)
    
            psnr_scores.append(psnr)
            original_niqe_scores.append(original_niqe)
            denoised_niqe_scores.append(denoised_niqe)
            original_piqe_scores.append(original_piqe)
            denoised_piqe_scores.append(denoised_piqe)
            original_brisque_scores.append(original_brisque)
            denoised_brisque_scores.append(denoised_brisque)
    
            results_data.append({
                'filename': filename,
                'psnr': psnr,
                'piqe': denoised_piqe,
                'original_piqe': original_piqe,
                'brisque': denoised_brisque,
                'original_brisque': original_brisque,
                'niqe': denoised_niqe,
                'original_niqe': original_niqe
            })
    
    # --- Save per-image metrics to CSV ---
    save_results_dir = os.path.join(opt.save_results_path, opt.get_hyperparameter_string())
    os.makedirs(save_results_dir, exist_ok=True)
    csv_path = os.path.join(save_results_dir, "results.csv")
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df[['filename', 'psnr', 'piqe', 'original_piqe',
                             'brisque', 'original_brisque', 'niqe', 'original_niqe']]
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Metrics saved to {csv_path}")
    
    # --- Save average metrics ---
    avg_metrics = {
        'run_name' : hyperparam_str,
        'avg_psnr': np.mean(psnr_scores),
        'avg_piqe': np.mean(denoised_piqe_scores),
        'avg_original_piqe': np.mean(original_piqe_scores),
        'avg_brisque': np.mean(denoised_brisque_scores),
        'avg_original_brisque': np.mean(original_brisque_scores),
        'avg_niqe': np.mean(denoised_niqe_scores),
        'avg_original_niqe': np.mean(original_niqe_scores),
    }
    
    # --- Append to or create general CSV ---
    all_runs_csv = os.path.join(opt.save_results_path, "avg_results.csv")
    
    if os.path.exists(all_runs_csv):
        df_all = pd.read_csv(all_runs_csv)
        df_all = pd.concat([df_all, pd.DataFrame([avg_metrics])], ignore_index=True)
    else:
        df_all = pd.DataFrame([avg_metrics])
    
    df_all.to_csv(all_runs_csv, index=False)
    print(f"\nAverage results appended to: {all_runs_csv}")
    
    # --- Save denoised images and residuals ---
    denoised_dir = os.path.join(save_results_dir, "denoised_images")
    residual_dir = os.path.join(save_results_dir, "residuals")
    os.makedirs(denoised_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)
    
    for i, filename in enumerate(results_df['filename']):
        denoised_uint8 = (np.clip(denoised_images_np[i], 0, 1) * 255).astype(np.uint8)
        residual_display = (residuals[i] - residuals[i].min()) / (np.ptp(residuals[i]) + 1e-8)
        residual_uint8 = (residual_display * 255).astype(np.uint8)
    
        Image.fromarray(denoised_uint8).save(os.path.join(denoised_dir, f"{filename}.png"))
        Image.fromarray(residual_uint8).save(os.path.join(residual_dir, f"{filename}_residual.png"))
    
    print("All denoised images and residuals saved successfully.")
    print("------------------------------------------------")

    print(f"--- Run {hyperparam_str} Finished ---")


if __name__ == "__main__":
    # --- Setup CUDA device and print info at the start ---
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available, using CPU.")
    
    parser = argparse.ArgumentParser(description="Neighbor2Neighbor (Nei2Nei) Denoising Training Script")
    
    # <<< FIX 1: Replaced N2S arguments with Nei2Nei arguments ---
    parser.add_argument('--n_epoch', type=int, help="Number of training epochs")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--batchsize', type=int, help="Batch size")
    parser.add_argument('--patchsize', type=int, help="Patch size (e.g., 512, 1024)")
    parser.add_argument('--n_feature', type=int, help="Number of features in UNet (e.g., 48)")
    parser.add_argument('--increase_ratio', type=float, help="Lambda increase ratio")
    
    args = parser.parse_args()
    
    # Create the config object from defaults
    opt = TrainingConfig()
    
    # Override defaults with any arguments that were provided
    for key, value in vars(args).items():
        if value is not None:
            setattr(opt, key, value)

    # This calls the method to set the correct img_dir and test_img_dir
    opt.update_paths() 
    
    # Run the main logic
    main(opt)
        