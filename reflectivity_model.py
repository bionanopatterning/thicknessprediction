import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

### ### ### ### ### ### ### SETTINGS ### ### ### ### ### ### ###
dMin = 0
dMax = 2001

lambda1 = 528
lambda2 = 470
lambda3 = 405
n = 1.25

wavelength_min = 300
wavelength_max = 800
wavelength_step = 1

SCATTER_LENGTH_SCALE = 1000
spectra_file_path = "Reflection_spectra.csv"

class Spectrum:
    LAMBDA_MIN = 300
    LAMBDA_MAX = 800
    LAMBDA_STEP = 1
    LAMBDA = np.linspace(LAMBDA_MIN, LAMBDA_MAX, int((LAMBDA_MAX - LAMBDA_MIN) / LAMBDA_STEP))

    def __init__(self, path, name=""):
        self.path = path
        self.data = dict()
        self.name = name
        data = pd.read_csv(path)
        data.fillna(0)
        header = data.head()
        self.wavelength = None
        for h in header:
            if 'avelength' in h:
                self.wavelength = data[h]
            else:
                self.data[h] = data[h]

        for channel in self.data:
            self.data[channel] = np.nan_to_num(self.data[channel])
            self.data[channel] = np.interp(Spectrum.LAMBDA, self.wavelength, self.data[channel])
            self.data[channel] /= np.amax(self.data[channel])

    def at(self, wavelength, channel):
        l = np.floor(wavelength)
        u = (wavelength - l)
        idx = np.abs(Spectrum.LAMBDA - l).argmin()
        try:
            return self.data[channel][idx] * (1 - u) + self.data[channel][idx + 1] * u
        except Exception as e:
            return np.nan

    def range(self, lmin, lmax, type, step=None):
        if not step:
            if not isinstance(lmin, int) or not isinstance(lmax, int):
                raise RuntimeError("lmin and lmax in Spectrum.range should be integers")
            elif lmin < Spectrum.LAMBDA_MIN:
                raise RuntimeError("lmin should be > 300")
            elif lmax > Spectrum.LAMBDA_MAX:
                raise RuntimeError("lmax should be < 800")
            elif not type in self.data:
                raise RuntimeError("This spectrum has no " + type + "component.")
            else:
                minindex = lmin - Spectrum.LAMBDA_MIN
                maxindex = lmax - Spectrum.LAMBDA_MIN
                return self.data[type][minindex:maxindex]
        else:
            lam = np.linspace(lmin, lmax, int((lmax - lmin) / step))
            spc = np.interp(lam, Spectrum.LAMBDA, self.data[type])
            return spc

    def plot(self, type=None, lims=None, **kwargs):
        if not lims:
            lims = (Spectrum.LAMBDA_MIN, Spectrum.LAMBDA_MAX)

        wavelength = np.linspace(lims[0], lims[1], lims[1] - lims[0])
        if not type:
            for component in self.data:
                if self.name:
                    labelstr = self.name + " - " + component
                else:
                    labelstr = component
                plt.plot(wavelength, self.range(lims[0], lims[1], component), label=labelstr, **kwargs)
        else:
            if self.name:
                labelstr = self.name + " - " + type
            else:
                labelstr = type
            plt.plot(wavelength, self.range(lims[0], lims[1], type), label=labelstr, **kwargs)
        plt.legend()

    def __str__(self):
        specStr = "Spectrum obj. repr. of file: " + self.path + " with components: \n"
        i = 1
        for type in self.data:
            specStr += "\t " + str(i) + "\t-\t" + type + "\n"
            i += 1
        return specStr


def _nrm(a):
    a -= np.amin(a)
    a /= np.amax(a)

spectra = Spectrum(spectra_file_path)
wavelength = np.linspace(wavelength_min, wavelength_max, wavelength_max - wavelength_min)

LED_R = spectra.range(wavelength_min, wavelength_max, "Omicron LedHUB 530nm LED", step = wavelength_step)
LED_G = spectra.range(wavelength_min, wavelength_max, "Omicron LedHUB 475nm LED", step = wavelength_step)
LED_B = spectra.range(wavelength_min, wavelength_max, "Omicron LedHUB 405nm LED", step = wavelength_step)
BPF_R = spectra.range(wavelength_min, wavelength_max, "Semrock FF01-532/18", step = wavelength_step)
BPF_G = spectra.range(wavelength_min, wavelength_max, "Semrock FF01-470/28", step = wavelength_step)
BPF_B = spectra.range(wavelength_min, wavelength_max, "Semrock FF01-400/40", step = wavelength_step)
QE = spectra.range(wavelength_min, wavelength_max, "pco.edge 4.2", step = wavelength_step)
DICHROIC = spectra.range(wavelength_min, wavelength_max, "Chroma 51000bs", step = wavelength_step)

spectrum_r = LED_R * BPF_R * (1.0 - DICHROIC) * DICHROIC * QE
spectrum_g = LED_G * BPF_G * (1.0 - DICHROIC) * DICHROIC * QE
spectrum_b = LED_B * BPF_B * (1.0 - DICHROIC) * DICHROIC * QE

_nrm(spectrum_r)
_nrm(spectrum_g)
_nrm(spectrum_b)

d = np.linspace(dMin, dMax, int((dMax - dMin)))

def reflectivity_from_spectrum(spectrum):
    reflectivity = np.zeros((len(d), len(wavelength)))
    for i in range(len(wavelength)):
        reflectivity[:, i] = np.cos(4 * np.pi * d * n / wavelength[i] + np.pi) * spectrum[i] * 0.5 * np.exp(-d / SCATTER_LENGTH_SCALE) + 0.5
    total_reflectivity = np.sum(reflectivity, axis = 1)
    total_reflectivity -= np.amin(total_reflectivity)
    total_reflectivity /= np.amax(total_reflectivity)
    return total_reflectivity

r = reflectivity_from_spectrum(spectrum_r)
g = reflectivity_from_spectrum(spectrum_g)
b = reflectivity_from_spectrum(spectrum_b)
a = np.maximum.accumulate(r)

def thickness_to_rgb(thickness):
    if thickness > dMax:
        print(f"thickness {thickness} is higher than model's maximum thickness. adjust parameter in reflectivity_model.py")
        raise Exception()
    idx = int((thickness - dMin))
    return (r[idx], g[idx], b[idx], a[idx])

def plot_model():
    plt.figure("Reflectivity model")
    plt.tight_layout()

    plt.subplot(3,1,1)
    plt.title("LED source spectra")
    plt.plot(wavelength, spectrum_r, color = (0.2, 0.8, 0.0), label = f"{lambda1} nm")
    plt.plot(wavelength, spectrum_g, color = (0.0, 0.8, 0.8), label = f"{lambda2} nm")
    plt.plot(wavelength, spectrum_b, color = (0.8, 0.0, 0.8), label = f"{lambda3} nm")
    plt.legend()
    plt.xlim([300, 700])


    plt.subplot(3,1,2)
    plt.plot(d, r, color = (0.8, 0.0, 0.0), linewidth = 2, label = f"{lambda1} nm")
    plt.plot(d, g, color = (0.0, 0.8, 0.0), linewidth = 2, label = f"{lambda2} nm")
    plt.plot(d, b, color = (0.0, 0.0, 0.8), linewidth = 2, label = f"{lambda3} nm")
    plt.title("Model - reflectivity as function of sample thickness")
    plt.legend()
    plt.xlim([dMin, dMax])


    plt.subplot(3,1,3)
    colorbar = np.zeros((1, len(r), 3))
    colorbar[:, :, 0] = r
    colorbar[:, :, 1] = g
    colorbar[:, :, 2] = b
    colorbar = np.resize(colorbar, (20, len(r), 3))
    plt.imshow(colorbar, aspect = "auto")
    plt.show()


if __name__ == "__main__":
    plot_model()