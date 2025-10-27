import subprocess
import itertools

# --- 1. Define the MAIN grids for 512x512 ---
# unet_grid_512 = {
#     'model': [1], # 1 = Unet
#     'n_epoch': [50, 75],
#     'lr': [0.001],
#     'batchsize': [8],
#     'patchsize': [512],
#     'masker_width': [4, 8, 16]
# }

# dncnn_grid_512 = {
#     'model': [0], # 0 = DnCNN
#     'num_layers': [8, 12],
#     'n_epoch': [50, 75],
#     'lr': [0.001],
#     'batchsize': [4],
#     'patchsize': [512],
#     'masker_width': [4, 8, 16]
# }

# --- 2. Define your SPECIAL 1024x1024 experiments ---
special_experiments = [
    {
        'model': 1, 'n_epoch': 25, 'lr': 0.001, 'batchsize': 2,
        'patchsize': 1024, 'masker_width': 8
    },
    {
        'model': 0, 'num_layers': 8, 'n_epoch': 25, 'lr': 0.001,
        'batchsize': 2, 'patchsize': 1024, 'masker_width': 8
    }
]

# --- 2A. NEW: Define Completed Experiments to Skip ---
# (Assuming ep5 meant n_epoch=50)
# completed_experiments_list = [
#     {
#         'model': 1, 'n_epoch': 50, 'lr': 0.001, 'batchsize': 8,
#         'patchsize': 512, 'masker_width': 4
#     },
#     {
#         'model': 1, 'n_epoch': 50, 'lr': 0.001, 'batchsize': 8,
#         'patchsize': 512, 'masker_width': 8
#     },
#     {
#         'model': 1, 'n_epoch': 50, 'lr': 0.001, 'batchsize': 8,
#         'patchsize': 512, 'masker_width': 16
#     }
# ]

# --- 3. Helper function (No change) ---
def create_experiments(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# --- 4. Create the final experiment list ---
# unet_experiments_512 = create_experiments(unet_grid_512)
#dncnn_experiments_512 = create_experiments(dncnn_grid_512)

all_experiments = special_experiments

# --- 4A. NEW: Filter out completed experiments ---
#all_experiments = []
# for exp in all_experiments_raw:
#     if exp not in completed_experiments_list:
#         all_experiments.append(exp)
#     else:
#         print(f"Skipping already completed experiment: {exp}")

#print(f"\nTotal experiments generated: {len(all_experiments_raw)}")
# print(f"Skipping {len(completed_experiments_list)} completed experiments.")
print(f"--- Total experiments TO RUN: {len(all_experiments)} ---")
#print(f"  ({len(unet_experiments_512) - 3} U-Net 512x512 experiments)") # 3 were skipped
#print(f"  ({len(dncnn_experiments_512)} DnCNN 512x512 experiments)")
print(f"  ({len(special_experiments)} Special 1024x1024 experiments)")


# --- 5. Loop and run each experiment (No change) ---
for i, params in enumerate(all_experiments):
    print(f"\n--- Starting Experiment {i+1} / {len(all_experiments)} ---")
    print(params)
    
    # Build the command-line command
    cmd = ['python', 'train.py']
    for key, value in params.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))
        
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! EXPERIMENT FAILED: {params} !!!")
        print(f"Return code: {e.returncode}")

print("\nAll experiments complete.")