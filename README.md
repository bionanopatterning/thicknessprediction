# thicknessprediction

This repository contains the code used for our paper "Predicting cryo-TEM sample thickness using reflected light microscopy" [...]. In the paper, we present a method for the prediction of cryo-TEM sample thickness based on a reflected light microscopy image of the sample. While we use a custom-built cryogenic fluroescence microscope for our experiments, it should be possible to acquire similar images on a regular wide-field, LED illuminated fluorescence microscope, provided a filter set that can be used for reflection imaging is available (as well as a cryogenic sample stage, in case of cryo imaging). The image below shows an example of a reflected light image alongside the corresponding sample thickness prediction and the measured ground truth.

![alt text](https://github.com/bionanopatterning/thicknessprediction/blob/master/readme_image.png "")

A brief guide to get the code running can be found below.

## Usage
To test the code, we recommend cloning this repository into PyCharm and running it within a virtual environment with the following packages installed (all of which are available on pypi from within PyCharm):
* numpy
* PIL
* colorcet
* pyplot
* pandas
* TensorFlow 2.8.0 (tensorflow-gpu)

We ran computations on a Quadro P2200 GPU, which in our case required installation of cuDNN version 8.1.0 and CUDA version 11.2. Our version of Python was 3.9.

Each of the .py files (model, simulator, reflectivity_model, and main) have settings at the top. To simulate height maps and corresponding reflectivity images, run simulator.py as the top-level script after adjusting the settings to your preference (e.g. the output folder). Paired training images will be saved into the output folder. Note that the simulator relies on the reflectivity model (reflectivity_model.py). The settings in this file can be changed to adjust the model to, e.g. another microscope with different light sources (in which case different spectra must be used; change the variable ' spectra_file_path' in reflectivity_model.py). The reflectivity model script can be ran as a top-level script to plot the reflectivity model.

After generating training data, the neural network can be trained by running the main script. The variable 'maxHeight' in the main script must be set to a value equal to or higher than the maximum sample thickness used to simulate the training data (the neural network operates on images scaled to within the 0.0 - 1.0 range; the max height value is the normalization factor). Depending on your hardware, training can take a while or be done within tens of minutes; in our case training for 5 epochs typically took about 90 minutes (fairly slow). After 5 epochs, the thickness predictions will have started to resemble the ground truth. A total training time 50 epochs is a good target.

After every epoch of training, a checkpoint of the model parameters is saved. To test the model on real data or on validation images, these checkpoints can be loaded. The commented block of code in main.py can be used to load a checkpoint, load and process a test image, and output the predicted thickness.

For comments or questions, please contact: m.g.f.last@lumc.nl

