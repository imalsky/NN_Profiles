Getting Started
===============

Installation
------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/atmos-transformer.git
      cd atmos-transformer

2. **Install dependencies**:

   .. code-block:: bash

      pip install -r requirements.txt

3. **Data and Inputs**:
   - Prepare or generate your input JSON profiles (``pressure``, ``temperature``, etc.)
   - Place them in the folder structure matching ``dataset.py`` usage (e.g. ``data/normalize_profiles``).

Quickstart
----------

Once installed, you can train the model using:

.. code-block:: bash

   python main.py --gen_profiles_bool=False --normalize_data_bool=False --create_model=True

This command:

- Reads the default parameters from ``model_input_params.json``.
- Trains the Transformer-based model on the normalized profile data.

``main.py`` supports other flags for:
- Generating new profiles (``--gen_profiles_bool=True``),
- Normalizing raw data (``--normalize_data_bool=True``),
- Hyperparameter tuning (``--create_and_hypertune=True``).

``main.py`` entrypoint
----------------------
Example run:

.. code-block:: bash

   python main.py

Will load default configuration and proceed with the pipeline steps you set in the code.

