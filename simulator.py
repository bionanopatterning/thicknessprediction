import numpy as np
import reflectivity_model
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import pyfastnoisesimd as fns
import matplotlib.pyplot as plt
import reflectivity_model

### ### ### ### ### ### ### SETTINGS ### ### ### ### ### ### ###

folder = "220701_ValidationData/"

N_IMAGES = 100

# Settings for the ranges of parameters used by the perlin noise generator.
# For every simulated image, a random value is drawn from these ranges. Note that the range for the frequency parameter is that of the logarithm of the parameter value, not the value itself.
FREQUENCY_LOG_RANGE = [-2.5, -3]
LACUNARITY_RANGE = [1.8, 2.5]
GAIN_RANGE = [0.3, 0.5]
PEAK_HEIGHT_RANGE = [500, 1000] # Thickness maps will be generated with a range of 0 to PEAK_HEIGHT


BACKGROUND_INTENSITY_LOW = 40
BACKGROUND_INTENSITY_HI = 140
BACKGROUND_STD = 0
BLUR_RADIUS = 2

# Cells tend to be large flat regions of low thickness with steep, high thickness blobs near the center (nucleus). To add this feature to the simulated images, some of the heightmaps will have
# the height 'remapped' according to the following function: f(h) = h * SLOPE_SCALE_FAC for h < SLOPE_SWITCH_HEIGHT, else f(h) = h - (1.0 - SLOPE_SCALE_FAC) * SLOPE_SWITCH_HEIGHT
# I.e. flattening the slope somewhat for regions where the thickness is below some inflection value, and offsetting the height for regions above the inflection value
# to ensure no discontinuous height jumps are generated. This remapping is not done for all simulated thickness maps, only for a fraction FRACTION_OF_DATASET_THAT_WILL_HAVE_SLOPE_REMAPPED
SLOPE_SWITCH_HEIGHT_RANGE = [300, 600]
SLOPE_SCALE_FAC_RANGE = [0.3, 0.6]
FRACTION_OF_DATASET_THAT_WILL_HAVE_SLOPE_REMAPPED = 0.5

# below is a value that doesn't need to be touched, probably. It sets a minimum average height per pixel in a simulated thickness map. If the mean height in a simulated map is below this value,
# a new thickness map is generated. Maps with low per-pixel averages are almost empty maps with a sparse few low peaks - i.e., they are unlike real samples, so we discard them.
MINIMUM_PER_PIXEL_AVG_THICKNESS = 60
def perlin_noise_img(octaves, frequency, lacunarity, gain):
    perlin = fns.Noise(seed=np.random.randint(2 ** 31), numWorkers=4)
    perlin.frequency = frequency
    perlin.noiseType = fns.NoiseType.PerlinFractal
    perlin.fractal.octaves = octaves
    perlin.fractal.lacunarity = lacunarity
    perlin.fractal.gain = gain
    perlin.perturb.perturnType = fns.PerturbType.NoPerturb
    return perlin.genAsGrid((256, 256))

def generate_heightmap():
    frequency = 10**np.random.uniform(FREQUENCY_LOG_RANGE[0], FREQUENCY_LOG_RANGE[1])
    lacunarity = np.random.uniform(LACUNARITY_RANGE[0], LACUNARITY_RANGE[1])
    gain = np.random.uniform(GAIN_RANGE[0], GAIN_RANGE[1])
    noise = perlin_noise_img(octaves=8, frequency=frequency, lacunarity=lacunarity, gain=gain)
    noise[noise < 0] = 0
    if np.amax(noise) <= 0:
        return generate_heightmap() # if all values < 0 (these noise values would be mapped to 0 nm thickness), just try agian
    else:
        peak_height = np.random.uniform(PEAK_HEIGHT_RANGE[0], PEAK_HEIGHT_RANGE[1])
        heightmap = noise * peak_height / np.amax(noise)
        if np.mean(heightmap) < MINIMUM_PER_PIXEL_AVG_THICKNESS:
            return generate_heightmap() # If avg. thickness < this value, the image is too empty so try agian
        else:
            return heightmap

def remap_heights(heightmap):
    switch_height = np.random.uniform(SLOPE_SWITCH_HEIGHT_RANGE[0], SLOPE_SWITCH_HEIGHT_RANGE[1])
    scale_fac = np.random.uniform(SLOPE_SCALE_FAC_RANGE[0], SLOPE_SCALE_FAC_RANGE[1])
    for x in range(256):
        for y in range(256):
            h = heightmap[x, y]
            if h < switch_height:
                h *= scale_fac
            else:
                h -= (1.0 - scale_fac) * switch_height
            heightmap[x, y] = h
    return heightmap

def heightmap_to_rgb(heightmap):
    reflection_image = np.zeros((256, 256, 4))
    for x in range(256):
        for y in range(256):
            rgba = reflectivity_model.thickness_to_rgb(heightmap[x, y])
            reflection_image[x, y, 0] = rgba[0]
            reflection_image[x, y, 1] = rgba[1]
            reflection_image[x, y, 2] = rgba[2]
            reflection_image[x, y, 3] = rgba[3]
    return reflection_image

def generate_background_img():
    image = np.random.normal(loc = np.random.uniform(BACKGROUND_INTENSITY_LOW, BACKGROUND_INTENSITY_HI) / 255, scale = BACKGROUND_STD / 255, size=(256,256,3))
    image = gaussian_filter(image, sigma = BLUR_RADIUS)
    return image

def generate_pair():
    thickness = generate_heightmap()
    if np.random.uniform(0, 1) > FRACTION_OF_DATASET_THAT_WILL_HAVE_SLOPE_REMAPPED:
        thickness = remap_heights(thickness)
    rgba = heightmap_to_rgb(thickness)
    reflection = np.zeros((256, 256, 3))
    background = generate_background_img()
    for x in range(256):
        for y in range(256):
            alpha = rgba[x, y, 3]
            reflection[x, y] = (1 - alpha) * background[x, y] + alpha * rgba[x, y, :3]
    reflection *= 255
    reflection = reflection.astype(np.uint8)

    return thickness, reflection

if __name__ == "__main__":
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(N_IMAGES):
        thickness, reflection = generate_pair()
        Image.fromarray(thickness).save(folder + f"00{i}_thickness.tiff")

        png = Image.fromarray(reflection).convert('RGB').save(folder + f"00{i}_reflection.png")