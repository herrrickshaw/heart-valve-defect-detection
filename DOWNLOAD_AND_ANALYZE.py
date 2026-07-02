"""
Download CAMUS Dataset from Kaggle and Analyze with Heart Disease System

Two methods:
1. kagglehub (modern, recommended)
2. kaggle CLI (traditional)
"""

import os
from pathlib import Path

# ============================================================================
# METHOD 1: Using kagglehub (Recommended - Simpler)
# ============================================================================

def download_camus_kagglehub():
    """
    Download CAMUS dataset using kagglehub

    Installation:
        pip install kagglehub

    Authentication:
        1. Go to https://www.kaggle.com/settings/account
        2. Click "Create New API Token"
        3. This creates ~/.kaggle/kaggle.json
        kagglehub will auto-detect it
    """
    import kagglehub

    print("Downloading CAMUS dataset via kagglehub...")

    # Download latest version
    dataset_path = kagglehub.dataset_download("parsakh/camus-echocardiography-image-dataset")

    print(f"✅ Dataset downloaded to: {dataset_path}")
    return Path(dataset_path)


# ============================================================================
# METHOD 2: Using kaggle CLI (Traditional)
# ============================================================================

def download_camus_kaggle_cli(output_dir: str = "./camus_data"):
    """
    Download CAMUS dataset using kaggle CLI

    Installation:
        pip install kaggle

    Authentication:
        1. Go to https://www.kaggle.com/settings/account
        2. Click "Create New API Token"
        3. Place kaggle.json in ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading CAMUS dataset via kaggle CLI to {output_dir}...")

    # Download
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "shoybhasan/camus-human-heart-data",
        "-p", output_dir
    ])

    # Extract
    zip_file = Path(output_dir) / "camus-human-heart-data.zip"
    if zip_file.exists():
        subprocess.run(["unzip", "-q", str(zip_file), "-d", output_dir])
        print(f"✅ Dataset extracted to: {output_dir}")

    return Path(output_dir)


# ============================================================================
# LOAD INTO YOUR HEART DISEASE ANALYSIS SYSTEM
# ============================================================================

def analyze_camus_dataset(dataset_path: Path):
    """
    Load CAMUS and run analysis with your heart disease system
    """
    from heart_valve_analysis import (
        CAMUSDataset,
        UNetSegmenter,
        SegmentationTrainer,
        SegmentationValidator
    )
    import torch
    from torch.utils.data import DataLoader

    print("\n" + "="*70)
    print("INITIALIZING HEART DISEASE ANALYSIS SYSTEM")
    print("="*70)

    # 1. Load dataset
    print("\n[STEP 1] Loading CAMUS dataset...")
    train_dataset = CAMUSDataset(split='train', target_shape=(256, 256))
    val_dataset = CAMUSDataset(split='val', target_shape=(256, 256))

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # 2. Create data loaders
    print("\n[STEP 2] Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 3. Initialize model
    print("\n[STEP 3] Initializing U-Net segmentation model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = UNetSegmenter(in_channels=1, num_classes=3)
    trainer = SegmentationTrainer(model, device=device, learning_rate=1e-4)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Training loop
    print("\n[STEP 4] Starting training...")
    print("  Target: Dice > 0.931 (CAMUS benchmark)")
    print("  X-error < 1.35 mm, Y-error < 0.75 mm")

    num_epochs = 50
    best_dice = 0

    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics = trainer.validate(val_loader)
            dice = val_metrics['dice_lv']

            # Check against benchmark
            benchmark = val_metrics['benchmark_lv_ed']
            status = "✅ ABOVE BENCHMARK" if dice > benchmark else "⚠️  Below benchmark"

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val Dice LV: {dice:.4f} ({benchmark:.4f} target) {status}")

            if dice > best_dice:
                best_dice = dice
                # Save checkpoint
                torch.save(model.state_dict(), f'model_epoch_{epoch+1}_dice_{dice:.4f}.pt')
                print(f"  💾 Saved checkpoint")

    # 5. Final validation
    print("\n[STEP 5] Final validation...")
    final_metrics = trainer.validate(val_loader)

    print("\nFINAL RESULTS:")
    print("="*70)
    print(f"Dice LV (Validation): {final_metrics['dice_lv']:.4f}")
    print(f"Dice LA (Validation): {final_metrics['dice_la']:.4f}")
    print(f"CAMUS Benchmark (LV ED): {final_metrics['benchmark_lv_ed']:.4f}")
    print(f"CAMUS Benchmark (LV ES): {final_metrics['benchmark_lv_es']:.4f}")

    if final_metrics['dice_lv'] > final_metrics['benchmark_lv_ed']:
        print("\n✅ SUCCESS: Model exceeds CAMUS benchmark!")
    else:
        print("\n⚠️  Model below benchmark. Consider:")
        print("  - Train for more epochs")
        print("  - Increase batch size")
        print("  - Use data augmentation")
        print("  - Check training data")

    print("="*70)

    return model, trainer, final_metrics


# ============================================================================
# QUICK ANALYSIS WITHOUT TRAINING
# ============================================================================

def quick_analyze_pretrained(model_path: str):
    """
    Load a pre-trained model and analyze CAMUS validation set
    """
    import torch
    from torch.utils.data import DataLoader
    from heart_valve_analysis import CAMUSDataset, UNetSegmenter, SegmentationValidator

    print("Loading pre-trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNetSegmenter(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load validation data
    val_dataset = CAMUSDataset(split='val', target_shape=(256, 256))
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Validate
    print("Running validation on CAMUS test set...")
    with torch.no_grad():
        predictions = []
        ground_truths = []

        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.softmax(logits, dim=1)
            predictions.append(preds.cpu().numpy())
            ground_truths.append(masks.numpy())

    # Compute metrics
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    metrics = SegmentationValidator.compute_metrics(
        predictions, ground_truths,
        pixel_spacing_x=0.30,
        pixel_spacing_y=0.15
    )

    print("\nVALIDATION METRICS:")
    print(f"Dice LV: {metrics['dice_lv_median']:.4f}")
    print(f"Dice LA: {metrics['dice_la_median']:.4f}")
    print(f"X-error: {metrics['x_error_median_mm']:.2f} mm")
    print(f"Y-error: {metrics['y_error_median_mm']:.2f} mm")

    return model, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    print("🏥 HEART DISEASE ANALYSIS SYSTEM - CAMUS DATASET DOWNLOAD & ANALYSIS")
    print("="*70 + "\n")

    # Choose download method
    method = input("Download method?\n1. kagglehub (recommended)\n2. kaggle CLI\nChoice (1-2): ").strip()

    try:
        if method == "1":
            print("\n→ Using kagglehub method...")
            dataset_path = download_camus_kagglehub()
        elif method == "2":
            print("\n→ Using kaggle CLI method...")
            dataset_path = download_camus_kaggle_cli()
        else:
            print("Invalid choice. Using kagglehub...")
            dataset_path = download_camus_kagglehub()

        # Analyze
        print("\nProceed with analysis?")
        choice = input("1. Train model (50 epochs, ~4-6 hours GPU)\n2. Quick validation only\n3. Exit\nChoice (1-3): ").strip()

        if choice == "1":
            print("\n⏳ This will take 4-6 hours on GPU...")
            model, trainer, metrics = analyze_camus_dataset(dataset_path)
            print(f"\n✅ Training complete! Model saved.")

        elif choice == "2":
            print("\nNote: Requires pre-trained model weights (.pt file)")
            model_path = input("Enter path to model weights: ").strip()
            if Path(model_path).exists():
                model, metrics = quick_analyze_pretrained(model_path)
            else:
                print("❌ Model file not found")

        print("\n✅ Analysis complete!")

    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nInstall with:")
        if method == "1":
            print("  pip install kagglehub torch torchvision")
        else:
            print("  pip install kaggle torch torchvision")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure kaggle.json is in ~/.kaggle/")
        print("2. Check internet connection")
        print("3. Verify CUDA/GPU setup if using GPU")


# ============================================================================
# STANDALONE USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Download with kagglehub
    python DOWNLOAD_AND_ANALYZE.py
    # Choose: 1
    # Choose: 1 (Train)

EXAMPLE 2: Use directly in Python
    from DOWNLOAD_AND_ANALYZE import download_camus_kagglehub, analyze_camus_dataset

    path = download_camus_kagglehub()
    model, trainer, metrics = analyze_camus_dataset(path)

EXAMPLE 3: Load CAMUS without download (if already downloaded)
    from heart_valve_analysis import CAMUSDataset
    from torch.utils.data import DataLoader

    dataset = CAMUSDataset(split='val')
    loader = DataLoader(dataset, batch_size=8)

    # Use with your model...

EXAMPLE 4: Download and analyze specific dataset version
    import kagglehub

    # List available versions
    versions = kagglehub.dataset_list_versions("parsakh/camus-echocardiography-image-dataset")

    # Download specific version
    path = kagglehub.dataset_download(
        "parsakh/camus-echocardiography-image-dataset",
        path=".",
        remote="origin",
        unzip=True
    )
"""
