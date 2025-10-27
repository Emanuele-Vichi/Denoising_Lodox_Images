#!/usr/bin/env python
# coding: utf-8

# # Noise2Self Denoising

# --- Core and Data Handling Libraries ---
import os
import time
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse

# --- PyTorch Libraries ---
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision import transforms

# --- Image Processing and Data Science Utilities ---
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import pyiqa

# --- Import from the noise2self repository ---
from mask import Masker
from models.unet import Unet
from models.dncnn import DnCNN
from util import show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingConfig:
    # --- Data and Augmentation ---
    val_split_ratio = 0.15

    # --- Model and Training Hyperparameters ---
    model = 1 # 0 = DnCNN, 1 = Unet
    n_epoch = 50
    n_channel = 1  # Set to 1 for grayscale X-ray images
    num_layers = 8 # For DnCNN
    lr = 0.001
    batchsize = 4
    patchsize = 512  # Can be 512, 1024, or 2048
    masker_width = 8

    # --- Paths ---
    save_model_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/N2S/Models'          # Base path to save model checkpoints
    save_losses_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/N2S/Losses'
    save_results_path = 'C:/Users/emanu/OneDrive - University of Cape Town/EEE4022S/Data/Final/N2S/Results'

    def __init__(self):
        self.update_paths()
    
    def get_hyperparameter_string(self):
        model_str = "Unet" if self.model == 1 else f"DnCNN{self.num_layers}"
        return (f"{model_str}_ep{self.n_epoch}_lr{self.lr}_"
                f"b{self.batchsize}_p{self.patchsize}_mw{self.masker_width}")

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
        print(f'Initialized dataset with {len(self.image_files)} images.')

    def __getitem__(self, index):
        img_path = self.image_files[index]
        im = Image.open(img_path).convert('L') # Convert to grayscale
        if self.transform:
            im = self.transform(im)
        return im, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_files)


def main(opt):
    """
    This function contains all the logic from your run_experiment function above.
    It takes the 'opt' object as an argument.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    # --- 2. Model Training ---
    print("Starting training...")
    # The Masker is central to Noise2Self, creating masks to hide pixels during training
    masker = Masker(width=opt.masker_width, mode='interpolate')
    # Define the model used
    if opt.model == 1:
        model = Unet(n_channel_in=opt.n_channel, n_channel_out=opt.n_channel).to(device)
    elif opt.model == 0:
        model = DnCNN(1, num_of_layers=opt.num_layers).to(device)
    # Define the loss function
    loss_function = MSELoss()
    # Define the optimiser
    optimizer = Adam(model.parameters(), lr=opt.lr)
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x * opt.n_epoch) for x in [0.2, 0.4, 0.6, 0.8]], gamma=0.5)
    
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
    
    # --- TRAINING & VALIDATION ---\n",
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, opt.n_epoch + 1):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch}/{opt.n_epoch}")
    
        # --- MODIFIED: Unpack image and filename from loader ---
        # We don't need the filename for training, so we use _
        for i, (noisy_image_batch, _) in enumerate(train_iterator):
            noisy_image = noisy_image_batch.to(device)
    
            # Apply the Noise2Self mask
            net_input, mask = masker.mask(noisy_image, i)
    
            # Get the model's prediction and calculate loss
            net_output = model(net_input)
            loss = loss_function(net_output * mask, noisy_image * mask)
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_train_loss += loss.item()
            train_iterator.set_postfix({'loss': loss.item()})
    
        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch}/{opt.n_epoch}")
        with torch.no_grad():
            # --- MODIFIED: Unpack image and filename from loader ---
            for i, (val_image_batch, _) in enumerate(val_iterator):
                val_image = val_image_batch.to(device)
    
                # Apply the Noise2Self mask for validation loss calculation
                net_input_val, mask_val = masker.mask(val_image, i)
    
                # Get model output and calculate loss on masked pixels
                net_output_val = model(net_input_val)
                val_loss = loss_function(net_output_val * mask_val, val_image * mask_val)
    
                epoch_val_loss += val_loss.item()
                val_iterator.set_postfix({'val_loss': val_loss.item()})
    
        scheduler.step()
    
        # --- Log and Save ---
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
    
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
    
        print(f"Epoch [{epoch}/{opt.n_epoch}] | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")
    
        # --- MODIFIED: Save Best Model with descriptive name ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_name = f"best_model.pth"
            best_model_path = os.path.join(save_model_dir, best_model_name)
            torch.save(model.state_dict(), best_model_path)
            # print(f'   -> New best model saved to {best_model_path}')
    
    # --- Save Final Model ---
    final_model_name = f"final_model.pth"
    final_model_path = os.path.join(save_model_dir, final_model_name)
    torch.save(model.state_dict(), final_model_path)
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
    
    # --- 3. Final Evaluation ---
    print("Starting evaluation...")
    # --- Setup ---
    # --- Create IQA metric models ---
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    piqe_metric = pyiqa.create_metric('piqe', device=device)
    brisque_metric = pyiqa.create_metric('brisque', device=device)
    
    # --- Load the Best Model ---
    model_path = os.path.join(save_model_dir, "best_model.pth")
    if opt.model == 1:
        model = Unet(n_channel_in=opt.n_channel, n_channel_out=opt.n_channel).to(device)
    else:
        model = DnCNN(1, num_of_layers=opt.num_layers).to(device)
    
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
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
            denoised_image_tensor = model(original_image_tensor).clamp(0, 1)
    
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
    
    parser = argparse.ArgumentParser(description="Noise2Self Denoising Training Script")
    
    # Add arguments for all the hyperparameters you want to change
    parser.add_argument('--model', type=int, help="0=DnCNN, 1=Unet")
    parser.add_argument('--n_epoch', type=int, help="Number of training epochs")
    parser.add_argument('--num_layers', type=int, help="Number of layers for DnCNN")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--batchsize', type=int, help="Batch size")
    parser.add_argument('--patchsize', type=int, help="Patch size (e.g., 512, 1024)")
    parser.add_argument('--masker_width', type=int, help="Width for the N2S masker")
    
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




