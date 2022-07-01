import numpy as np
import reflectivity_model
from PIL import Image
import os

### ### ### ### ### ### ### ### SETTINGS ### ### ### ### ### ### ### ###

folder = "trainData_B_1000nm/"

N_IMAGES = 500
IMG_WIDTH = 256 # Model works on 256 x 256 imgs.
IMG_HEIGHT = 256

OCTAVES = 6
PERSISTENCE_MAX = 0.5
PERSISTENCE_MIN = 0.2

PEAK_RANGE = [500, 1000]
ZERO_OFFSET = 200

# output channels:
R = True
G = True
B = True


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

if not os.path.exists(folder):
    os.makedirs(folder)

_imgs_saved = 0
def generate_perlin_noise_2d(shape, res):
    # source : https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def gen_perlin_texture(shape, octaves=1, persistence=0.5):
    # source : https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (int(frequency), int(frequency)))
        frequency *= 2
        amplitude *= persistence
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return noise


def gen_img():
    noise = gen_perlin_texture((IMG_WIDTH, IMG_HEIGHT), octaves=OCTAVES, persistence=np.random.uniform(low=PERSISTENCE_MIN, high=PERSISTENCE_MAX))
    noise *= (np.random.uniform(low=PEAK_RANGE[0], high=PEAK_RANGE[1]) + ZERO_OFFSET - reflectivity_model.dMin)
    noise -= (ZERO_OFFSET - reflectivity_model.dMin)
    max_height = np.amax(noise)
    noise /= max_height
    noise = noise * noise * np.sign(noise)
    noise *= max_height
    noise[noise < reflectivity_model.dMin] = reflectivity_model.dMin


    rgb_image = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            rgbval = reflectivity_model.thickness_to_rgb(noise[x, y])
            rgb_image[x, y, 0] = rgbval[0]
            rgb_image[x, y, 1] = rgbval[1]
            rgb_image[x, y, 2] = rgbval[2]
    rgb_image[:, :, 0] -= np.amin(rgb_image[:, :, 0])
    rgb_image[:, :, 0] /= np.amax(rgb_image[:, :, 0])
    rgb_image[:, :, 1] -= np.amin(rgb_image[:, :, 1])
    rgb_image[:, :, 1] /= np.amax(rgb_image[:, :, 1])
    rgb_image[:, :, 2] -= np.amin(rgb_image[:, :, 2])
    rgb_image[:, :, 2] /= np.amax(rgb_image[:, :, 2])
    rgb_image *= 255

    if not R:
        rgb_image[:, :, 0] = 0
    if not G:
        rgb_image[:, :, 1] = 0
    if not B:
        rgb_image[:, :, 2] = 0
    return noise, rgb_image.astype(int)


def save_tiff(img):
    image = Image.fromarray(img)
    image.save(folder + f"00{_imgs_saved}_height.tiff")

def save_png(rgb):
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.convert('RGB')
    img.save(folder + f"00{_imgs_saved}_rgb.png")

def save_pair(heightmap, rgb):
    global _imgs_saved
    save_tiff(heightmap)
    save_png(rgb)
    _imgs_saved += 1

def heightmap_to_rgb(heightmap):
    x, y = np.shape(heightmap)
    rgbimg = np.zeros((x, y, 3))
    for i in range(x):
        for j in range(y):
            rgbimg[i, j, :] = reflectivity_model.thickness_to_rgb(heightmap[i, j])
            rgbimg[i, j, :] *= 255
    rgbimg = rgbimg.astype(np.uint8)
    return rgbimg

if __name__ == "__main__":
    for i in range(N_IMAGES):
        heightmap, rgb = gen_img()
        save_pair(heightmap, rgb)

