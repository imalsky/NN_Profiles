# NN_Profiles

NN_Profiles is a Python-based pipeline for training and evaluating physics-informed Recurrent Neural Networks (RNNs) on planetary atmospheric profiles. This repository supports data preprocessing, normalization, training, and visualization of predictions.

## Features

- **Data Preprocessing**: Normalize raw atmospheric profiles for model training.
- **Flexible Model Training**: Train RNN models with options for hyperparameter tuning.
- **Visualization Tools**: Generate interpretable plots comparing predictions and actual values.
- **Customizable Options**: Modify normalization techniques, loss functions

## File Structure


    NNN_Profiles
    ├── Data/                       # Ignored on github (big files)
    │   ├── Profiles/               # Raw input profiles in JSON format.
    │   ├── Normalized_Profiles/    # Preprocessed and normalized profiles.
    │   ├── Model/                  # Directory for saving trained models.
    │   ├── Opacities               # Dir with the different opacities used in the model training
    │
    ├── Figures/                    # Directory for saving prediction plots (Ignored).
    │
    ├── Inputs/                     # Input configuration files for simulation parameters.
    │   ├── parameters.json         # Configuration file with planet and simulation parameters.
    │
    │
    ├── calculate_fluxes.py          # Functions for computing heating rates and fluxes. 
    ├── calculate_opacities.py       # Functions for generating opacity data.
    ├── create_training.py           # Script to generate training datasets.
    ├── dataset.py                   # Dataset and DataLoader definitions.
    ├── main.py                      # Main pipeline for training, testing, and visualization.
    ├── models.py                    # Defines the RNN architectures and other models.
    ├── normalize.py                 # Script for preprocessing and normalizing the data.
    ├── plot_atm_structure.ipynb     # Visualization atmospheres
    ├── plot_model_predictions.ipynb # Jupyter Notebook for visualization.
    ├── pt_profile_generator.py      # Script for generating PT profiles.
    ├── README.md                    # Documentation for the project.
    ├── run_sbatch_gattaca.sh        # Sbatch CPU command Gattaca
    ├── run_sbatch_gattaca.sh        # Sbatch GPU command Gattaca 
    ├── train.py                     # Training routines for the models.
    └── utils.py                     # Utility functions for loading metadata and auxiliary tasks.

## Workflow

### 1. Data Preparation

Create initial profiles with python training.py. This will create TP profiles randomly drawn with a 6 parameter structure.
This always has a fixed planet radius, composition, and many other params. Eventually these params should be varied, and saved as well to be trained on.
Raw profiles should be placed in the `Data/Profiles` directory as JSON files. Each profile must contain:
- `pressure`: Array of pressure values.
- `temperature`: Array of temperature values.
- `net_flux`: Array of net flux values.

### 2. Data Normalization

Normalize and preprocess data with: python normalize_data.py

- Outputs are saved in `Data/Normalized_Profiles/`.
- Normalization metadata is saved as `Data/Normalized_Profiles/normalization_metadata.json`.

**Normalization Techniques**:
- **Pressure**: Log-scaled and normalized to [0, 1].
- **Temperature**: Standardized (mean = 0, std = 1).
- **Net Flux**: Fourth root transformation followed by standardization (mean = 0, std = 1).

### 3. Train the RNN

Train the model with: python main.py

**Default Training Configuration**:
- **Model**: LSTM-based RNN.
- **Input Features**: Temperature and pressure.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam with a learning rate of 1e-4.

The trained model is saved in `Data/Model/`.

### 4. Visualization

Generate plots for model predictions: python main.py --visualize_only

This will load the best saved model and visualize the predictions against actual values, using data from `Data/Normalized_Profiles/`.

## Future Work

- DISORT radiative transfer
- Varying planet compositions
- Add clouds
- Add hazes
- Add different premixed tables
- Add info about the C/O ratio and metallicity of the table to the metadata that the model is trained onRebuild trigger
Force refresh
