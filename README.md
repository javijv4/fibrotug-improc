# fibrotug-improc

# DSP Image Processing

# Installation using VScode

1. using the terminal, navigate to the location where you want to have the code and `do git clone https://github.com/javijv4/fibrotug-improc.git` . This will create a subfolder `fibrotug-improc/`
2. Open VScode, go to File → Open… → and select the `fibrotug-improc/` folder.
3. Do `cmd+shift+P` and type Python. Select Python: Create Environment…. Select Venv. It might ask you for the Python interpreter, select any that is > 3.9. 
4. Do `cmd+shift+P` and type terminal. Select Terminal: Create New Terminal.
5. Check that the Python interpreter is correct by typing `which python`. It should show you `path_to_fibrotug-improc/.venv/bin/python`
6. In the terminal, run `python -m pip install -e .` .This will install all the codes and required packages.
7. That’s it. It should be ready to run. You can test that everything works running the `register_pre_to_post.py` file that, as default, points to a test case that is downloaded when you clone the repository (`test_data/` folder)

# Processing images

WAIT! Before start, check that the post and pre files are in the same orientation (they can be flipped in 180)

1. Generate tissue masks: 
    1. `generate_tissue_masks.py` → `fibrotug_mask_init.tif`
        It works for both fibers and actinin
        - pre images: use actinin
        - post images: use fibers
    2. use itksnap to fixed up the mask.→ `fibrotug_mask.tif`
2. Register pre-to-post:
    1. `register_pre_to_post.py` → `fibrotug_mask_init.tif`
    2. open `pre_mask.tif` in itksnap and get rid of the posts →`pre_tissue_mask.tif`
3. Process Actin Images
    1. `actin_processing.py` → `improc_”which”_actin.npz`
        - `improc_”which”_actin.npz` contains the image angles and smooth angles.
    2. Visualization can be done with `visualize_actin.py`
4. Process Fiber Images