Usage Guide
===========

Data Generation
---------------
If you need to create synthetic radiative profiles, set:

.. code-block:: python

   main(gen_profiles_bool=True)

This calls ``create_training.gen_profiles`` to generate JSON files in ``data/profiles``.

Normalization
-------------
Set:

.. code-block:: python

   main(normalize_data_bool=True)

This triggers:
- ``calculate_global_stats`` to compute min/max or robust stats.
- ``process_profiles`` to apply scaling and write normalized profiles to ``data/normalize_profiles``.

Training
--------
To train with default config:

.. code-block:: python

   main(create_model=True)

Internally, it calls ``train_model_from_config``, which:
- Initializes the dataset via ``NormalizedProfilesDataset``.
- Splits into train/val/test sets (70/15/15).
- Creates an ``AtmosphericModel`` with parameters from JSON.
- Trains the model using:
- SmoothL1Loss (with `beta=1.0` default).
- AdamW optimizer.
- CosineAnnealingWarmRestarts scheduler.
- Saves the best model to ``data/model/best_model.pth``.

Hyperparameter Tuning
---------------------
Set:

.. code-block:: python

   main(create_and_hypertune=True)

This runs an Optuna study with 20 trials by default, modifying hyperparameters like:
- `d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward`, etc.

Advanced Options
----------------------
- **smooth_l1_beta**: controls outlier sensitivity in SmoothL1Loss.
- **scheduler** hyperparams (`T_0`, `T_mult`, `eta_min`) can be tuned in the objective function.
- **Alternative normalizations**: standard scaling or log scaling might be tested by adjusting the code in `normalize.py`.

