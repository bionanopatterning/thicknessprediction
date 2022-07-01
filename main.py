
from PIL import Image, ImageFilter
import model
import os
import datetime
import numpy as np
#import colorcet as cc
import matplotlib.pyplot as plt
#import simulator
import glob

if __name__ == "__main__":
    def process_img(image):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = image[:image.rfind(".")]+"_heightmap_RGB"
        out_path += "_"+timestamp+".tiff"
        heightmap, rgbimg = model.rgb_to_height_pathinput(image)
        Image.fromarray(heightmap).save(out_path)
        plt.subplot(1,2,1)
        plt.imshow(rgbimg)
        plt.subplot(1,2,2)
        plt.imshow(heightmap)
        plt.show()

    def process_folder(path):
        imgpaths = glob.glob(path+"*.png")
        for img in imgpaths:
            heightmap, rgbimg = model.rgb_to_height_pathinput(img)
            Image.fromarray(heightmap).save(img[:-4]+"_HEIGHTMAP.tiff")

    def validate_folder(path):
        imgpaths = glob.glob(path+"*.png")
        for img in imgpaths:
            prediction, rgbimg = model.rgb_to_height_pathinput(img)
            truth = Image.open(img[:-14]+"thickness.tiff")
            error = truth - prediction
            Image.fromarray(prediction).save(img[:-14]+"prediction.tiff")
            Image.fromarray(error).save(img[:-14]+"error.tiff")

    model.load_model("220531_RGB_Model/checkpoints/")
    validate_folder("C:/Users/mgflast/PycharmProjects/HeightmapTensorflow/220701_ValidationData/")

    ### EXAMPLE: TRAINING A NEW MODEL ###
    # ### ### ### ### ### ### ### ### SETTINGS ### ### ### ### ### ### ### ###
    # experiment_title = "220602_RGB_Model/"
    # training_data = "220602_RGB_Traindata/"
    # ### ### ### ### ### ### ### ### ######## ### ### ### ### ### ### ### ###
    # if not os.path.exists(experiment_title):
    #     os.mkdir(experiment_title)
    #
    # model.checkpoint_dir = experiment_title + "checkpoints/"
    # model.train_progress_folder = experiment_title + "progress/"
    #
    # if not os.path.exists(model.checkpoint_dir):
    #     os.mkdir(model.checkpoint_dir)
    # if not os.path.exists(model.train_progress_folder):
    #     os.mkdir(model.train_progress_folder)
    #
    # model.train_model(training_data, epochs = 50)