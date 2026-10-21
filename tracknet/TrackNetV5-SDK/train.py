import argparse
import torch
torch.backends.cudnn.benchmark = True
# from torch.optim import lr_scheduler
import torch.optim.lr_scheduler as pt_lr_scheduler
from torch.utils.data import DataLoader
import importlib.util
from pathlib import Path
import sys # make sure sys is imported
 
#torch.autograd.set_detect_anomaly(True)

# --- 1. Import the build functions of all our "factories" ---
# Import the top-level packages; the corresponding __init__.py files ensure all modules are registered
import models_factory
import datasets_factory
import losses_factory
import optimizers_factory
import metrics_factory
import engine

# Import our final "executor", the Runner
from engine.runner import Runner

def load_config_from_path(config_path: str):
    """Load a config module from a .py file path."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    # Load the .py file as a module
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    
    # [Important] Make sure the module can be loaded again (if Python has already cached a module with the same name)
    if config_path.stem in sys.modules:
        del sys.modules[config_path.stem]
        
    spec.loader.exec_module(cfg_module)
    return cfg_module

# ===================================================================
# Step 1: Refactor your original main() function into run_experiment()
# ===================================================================
def run_experiment(config_path_str: str):
    """
    Run one complete training using a single config file.
    (This is the body of your original main() function)
    """
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 [STARTING] Experiment: {config_path_str}")
        print(f"{'='*80}\n")
        
        # --- A. Load config ---
        # No longer hard-coded; use the passed-in argument instead
        cfg = load_config_from_path(config_path_str)
        print("✅ Configuration loaded successfully.")
        
        # --- B. Environment setup (handled inside the Runner or set here) ---
        # (unchanged)
        
        # --- C. Use the factories to build all components from the blueprint ---
        print("Building components from config...")
        
        # Build the model
        model = models_factory.build_model(cfg.model)
        print("✅ Model built successfully.")

        # Build the datasets
        train_dataset = datasets_factory.build_dataset(cfg.data['train'])
        val_dataset = datasets_factory.build_dataset(cfg.data['val'])
        print("✅ Datasets built successfully.")
        
        # Build the data loaders (DataLoader)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
        print("✅ DataLoaders built successfully.")

        # Build the loss function
        criterion = losses_factory.build_loss(cfg.loss)
        print("✅ Loss function built successfully.")

        # Build the optimizer
        optimizer = optimizers_factory.build_optimizer(model, cfg.optimizer)
        print("✅ Optimizer built successfully.")
        
        # Build the evaluation metric
        metric = metrics_factory.build_metric(cfg.evaluation['metric'])
        print("✅ Metric built successfully.")

        # Build the hooks (Hooks)
        hooks = engine.build_hooks(cfg.log_config['hooks'])
        if hasattr(cfg, 'custom_hooks'):
            hooks.extend(engine.build_hooks(cfg.custom_hooks))
        print("✅ All Hooks built successfully.")

        # Build the learning rate scheduler
        scheduler = None
        if hasattr(cfg, 'lr_config') and cfg.lr_config is not None:
            policy = cfg.lr_config.get('policy', None)
            
            if policy == 'Step':
                if 'step' not in cfg.lr_config:
                    raise ValueError("Step policy requires 'step'(list of milestones in lr_config")
                scheduler = pt_lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=cfg.lr_config['step'],
                    gamma=cfg.lr_config.get('gamma', 0.1)
                )
                print(f"✅ LR MultiStepLR scheduler built successfully.")

            elif policy == 'CosineAnnealing':
                scheduler = pt_lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.total_epochs - cfg.lr_config.get('warmup_iters', 0),
                    eta_min=cfg.lr_config.get('min_lr', 0)
                )
                print(f"✅ LR CosineAnnealingLR scheduler built successfully.")
            else:
                print(f"⚠️ LR policy '{policy}' is not supported yet or is None. Running with a fixed learning rate.")
                scheduler = None
        else:
            print("ℹ️ No LR scheduler configured. Running with a fixed learning rate.")

        # --- D. Instantiate the "race driver" (Runner) ---
        runner = Runner(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_scheduler=scheduler,
            hooks=hooks,
            cfg=cfg,
            # IsAMP=True
        )
        
        # --- E. Start training! ---
        runner.run()

        print(f"\n{'-'*80}")
        print(f"✅ [SUCCESS] Experiment {config_path_str} completed!")
        print(f"{'-'*80}\n")

    except Exception as e:
        # Add error handling so that if one experiment fails, the next experiment in the queue can continue
        print(f"\n{'!'*80}")
        print(f"❌ [FAILED] Experiment {config_path_str} encountered an error: {e}")
        import traceback
        traceback.print_exc() # Print the full error stack trace
        print(f"{'!'*80}\n")
    
    # [Important] Free GPU memory to prepare for the next experiment
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================================
# Step 2: Create a new main() function as the "experiment launcher"
# ===================================================================
def main():
    """
    New main function: scan configs, show a menu, and run the selected experiments in order.
    """
    config_dir = Path('./configs/')
    
    # 1. Automatically scan all .py files under configs/official
    print(f"🔍 Scanning {config_dir}...")
    # Use sorted to ensure the order is the same on every run
    config_paths = sorted([p for p in config_dir.glob('**/*.py') if p.name != '__init__.py'])
    
    if not config_paths:
        print(f"❌ Error: No .py config files found in {config_dir}.")
        return

    # 2. Show a menu
    print("\n--- 📜 Available config files ---")
    for i, path in enumerate(config_paths, 1):
        # Use :>2 to right-align the index so it looks nicer
        print(f"  {i:>2}: {path.name}")
    print("---------------------------\n")

    # 3. Let you enter the indices you want to run
    while True:
        selection_str = input("👉 Enter the config indices to run (separated by spaces, e.g. '1 3 2'): ")
        selected_indices = selection_str.split()
        
        experiments_to_run = [] # Stores (path_str, name)
        valid_input = True
        
        if not selected_indices:
            print("⚠️ Input is empty, please select at least one.")
            continue
            
        try:
            for s_idx in selected_indices:
                idx = int(s_idx)
                if 1 <= idx <= len(config_paths):
                    # Indices start from 1, the list starts from 0
                    selected_path = config_paths[idx - 1]
                    # Store the config's full path string and file name
                    experiments_to_run.append((str(selected_path), selected_path.name))
                else:
                    print(f"❌ Index '{idx}' is out of range (must be between 1 and {len(config_paths)}).")
                    valid_input = False
                    break # Stop parsing this batch of input
        except ValueError:
            print(f"❌ Invalid input: '{s_idx}' is not a number.")
            valid_input = False
        
        if valid_input:
            # Confirm the selection
            print("\nYou have selected the following run order:")
            for i, (_, name) in enumerate(experiments_to_run, 1):
                print(f"  {i}. {name}")
            confirm = input("Confirm execution? (y/n): ").strip().lower()
            if confirm == 'y':
                break # Input is valid and confirmed, break out of the 'while True' loop
            else:
                print("🔄 Cancelled, please enter again.")
        else:
            print("🔄 Please enter again.")
            
    # 4. Run the selected experiments in order
    print(f"\n✨ Preparing to run {len(experiments_to_run)} experiment(s) in order...")
    for i, (path_str, name) in enumerate(experiments_to_run, 1):
        print(f"\n--- Queue: {i} / {len(experiments_to_run)} ---")
        # Call the function we refactored
        run_experiment(path_str)
        
    print("\n🎉 All selected experiments have finished!")


if __name__ == '__main__':
    main() # Run the new launcher