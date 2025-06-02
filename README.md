```markdown
# KaggleML Particle Segmentation Project

This repository uses StarDist2D and FastAI to segment particles in electron microscopy images. The project is still under development and not yet complete.

## Project Structure

```

KaggleML/
├── train.py        # Training script: data preparation and StarDist2D model training
├── infer.py        # Inference script: load trained weights, segment new images, save results
├── README.md       # Project description (this file)
└── .gitignore      # Files and folders to ignore in Git

````

## Dependencies

- Python 3.8 or higher  
- TensorFlow (or PyTorch backend for StarDist)  
- fastai  
- stardist  
- csbdeep  
- scikit-learn  
- numpy  
- tifffile  
- scikit-image  
- pandas  
- matplotlib  

You can install the main packages with:
```bash
pip install fastai stardist csbdeep scikit-learn numpy tifffile scikit-image pandas matplotlib
````

## Usage

### 1. Data Preparation & Training

Place the Kaggle dataset under `/kaggle/input/electron-microscopy-particle-segmentation` (or adjust paths).
In the project root, run:

```bash
python train.py \
  --input_root "/kaggle/input/electron-microscopy-particle-segmentation" \
  --output_dir "./models" \
  --epochs 100 \
  --batch_size 4 \
  --patch_size 256 256 \
  --n_rays 32
```

* `--input_root`: Directory containing images and masks
* `--output_dir`: Where trained weights and exported models will be saved
* Other parameters can be adjusted as needed

### 2. Inference

After training, use `infer.py` to segment new images:

```bash
python infer.py \
  --model_dir "./models/stardist_run1" \
  --image_path "/path/to/gray_image.tif" \
  --output_mask "./results/labels.tif" \
  --output_csv "./results/coords.csv"
```

* `--model_dir`: Directory containing `weights_best.h5`
* `--image_path`: Path to the grayscale TIFF for inference
* `--output_mask`: Path to save the instance label TIFF
* `--output_csv`: Path to save centroid coordinates CSV

## Project Status

* Data loading and preprocessing: mostly complete
* Training script (`train.py`): functional
* Inference script (`infer.py`): basic functionality implemented
* **Planned Improvements**:

  * Automated threshold tuning
  * Batch inference and visualization
  * Config-driven workflow management
  * Unit tests and CI integration

> **Note**: This project is a work in progress. Some parameters and paths must be adjusted manually. Feel free to open issues or submit pull requests on GitHub for improvements.

## Contact

If you have questions or suggestions, please open an issue or contact:

* GitHub: [WYR186](https://github.com/WYR186)

