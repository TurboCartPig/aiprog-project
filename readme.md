# Final project of AI Programming at NTNU

## Setup and run

This project uses poetry for dependency management, and you will need
to interface with the project via the poetry command line interface.
This project also strictly depends on python 3.8. No other version has
been tested, and later ones are incompatible as of now.

Poetry can be installed with pip list this:
`pip install poetry`

To setup the project initially, run:
1. `poetry env use <path to your python38 install>`
2. `poetry install`
3. `export TF_FORCE_GPU_ALLOW_GROWTH="true"`
4. Run test_tf_gpu.py to see if CUDA is setup correctly
5. Run individual models and see results

To run the CNN model:
`poetry run python ./cnn.py`
