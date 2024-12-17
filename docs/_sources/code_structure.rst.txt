Code Structure
==============

This repository is organized around four main Python modules plus auxiliary scripts:

1. **main.py**:
    - The central orchestration file containing a ``main`` function that:
    - Loads config parameters from ``inputs/model_input_params.json``.
    - Generates radiative profiles (if requested).
    - Normalizes data.
    - Trains (or hyper-tunes) the Transformer model.

2. **dataset.py**:
   - Defines the ``NormalizedProfilesDataset`` for reading JSON profile files.
   - Handles both list and scalar variable expansions, ensuring consistent tensor shapes.

3. **transformer_model.py**:
   - Contains the ``AtmosphericModel`` Transformer architecture.
   - Optionally includes a residual MLP block after each TransformerEncoder layer.

4. **train.py**:
   - Implements the training loop, validation, early stopping, and hyperparameter tuning utilities.
   - Integrates with a scheduler (e.g., CosineAnnealingWarmRestarts) and a chosen loss function (e.g., SmoothL1Loss).

Supporting Files
----------------

- **inputs/model_input_params.json**: Default hyperparameters (batch size, epochs, learning rate, etc.).
- **inputs/parameters.json**: Additional configuration for generating or normalizing data.
- **data/**: Folder containing subfolders for raw profiles, normalized profiles, and trained models.

Typical Execution Flow
----------------------
1. ``main.py`` calls:
   - ``gen_profiles`` to create training data (if needed).
   - ``calculate_global_stats`` and ``process_profiles`` to normalize the raw data.
   - ``train_model_from_config`` to train the Transformer or perform hyperparameter tuning.

2. ``train_model_from_config``:
   - Builds an instance of ``AtmosphericModel`` using parameters from JSON config or an Optuna trial.
   - Initializes the dataset from ``NormalizedProfilesDataset``, splits into train/val/test.
   - Runs the training loop via ``train_model``.

3. ``transformer_model.py``:
    - The Transformer-based architecture is loaded. It includes:
    - Input projection (``nx -> d_model``)
    - Stacked encoder layers
    - Optional residual MLP blocks
    - Output projection to match ``ny``

Usage in Practice
-----------------
- Modify ``model_input_params.json`` or pass in new hyperparameters during tuning.
- Run ``main.py`` to orchestrate the entire pipeline from data generation to training.
- Model checkpoints are saved to ``data/model``.
- Logs and potential stats files remain in your working directory or the specified output folder.
