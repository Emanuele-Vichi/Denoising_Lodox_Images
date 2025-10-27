import subprocess
import itertools

# --- 1. Define the MAIN grids for 512x512 ---
# These will generate the bulk of your experiments
# nei2nei_grid_512 = {
#     'n_epoch': [50],
#     'lr': [3e-4, 1e-4],
#     'batchsize': [8],
#     'patchsize': [512],
#     'n_feature': [48, 64],
#     'increase_ratio': [0.5, 1.0, 2.0]
# }

# --- 2. Define your SPECIAL 1024x1024 experiments ---
# Manually add any 1024 runs here with a smaller batchsize
special_experiments_1024 = [
    {
        'n_epoch': 25,
        'lr': 3e-4,
        'batchsize': 2,       # <-- Smaller batch for 1024
        'patchsize': 1024,
        'n_feature': 48,
        'increase_ratio': 1.0
    },
    # You can copy/paste this block to add more 1024 runs
]

# --- 3. Define Completed Experiments to Skip ---
# Add any runs you've already finished here
completed_experiments_list = [
    # Example:
    # {
    #     'n_epoch': 50, 'lr': 3e-4, 'batchsize': 8, 'patchsize': 512,
    #     'n_feature': 48, 'increase_ratio': 1.0
    # },
]

# --- 4. Helper function to create combinations ---
def create_experiments(grid):
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# --- 5. Create the final experiment list ---
# main_experiments = create_experiments(nei2nei_grid_512)

# Combine all experiment lists
all_experiments_raw = special_experiments_1024

# Filter out completed experiments
all_experiments = []
for exp in all_experiments_raw:
    if exp not in completed_experiments_list:
        all_experiments.append(exp)
    else:
        print(f"Skipping already completed experiment: {exp}")

print(f"\nTotal experiments generated: {len(all_experiments_raw)}")
print(f"Skipping {len(completed_experiments_list)} completed experiments.")
print(f"--- Total experiments TO RUN: {len(all_experiments)} ---")

# --- 6. Loop and run each experiment ---
for i, params in enumerate(all_experiments):
    print(f"\n--- Starting Experiment {i+1} / {len(all_experiments)} ---")
    print(params)
    
    # Build the command-line command
    # --- IMPORTANT: Make sure this points to your new script ---
    cmd = ['python', 'train_n2n.py'] 
    
    for key, value in params.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))
        
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! EXPERIMENT FAILED: {params} !!!")
        print(f"Return code: {e.returncode}")

print("\nAll experiments complete.")