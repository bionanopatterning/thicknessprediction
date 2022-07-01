
from PIL import Image
import model
import os

if __name__ == "__main__":
    experiment_title = "experiment_SingleChannel_470/"
    training_data = "trainData_G_1000nm/"
    maxHeight = 1000.0

    if not os.path.exists(experiment_title):
        os.mkdir(experiment_title)

    model.checkpoint_dir = experiment_title + "checkpoints/"
    model.train_progress_folder = experiment_title + "progress/"
    model.HEIGHT_RESCALE = 1 / float(maxHeight)

    if not os.path.exists(model.checkpoint_dir):
        os.mkdir(model.checkpoint_dir)
    if not os.path.exists(model.train_progress_folder):
        os.mkdir(model.train_progress_folder)

    #train.train_model(training_data, 490, epochs = 5)
    model.train_from_checkpoint(training_data, buffersize = 490, epochs = 5, checkpoint = model.checkpoint_dir)

    def process_image(model, image, savepath):
        model.load_model(model)
        prediction = model.rgb_to_height(image)
        Image.fromarray(prediction).save(savepath)
    # model = r"C:\Users\mgflast\PycharmProjects\HeightmapTensorflow\experiment_SingleChannel_528_increasedPersistence\checkpoints"
    # image = r"W:\mgflast\10. Reflection and thickness\Figure4\option7\R_256.png"
    # savepath = r"W:\mgflast\10. Reflection and thickness\Figure4\option7\R_256_prediction.tiff"
    # process_image(model, image, savepath)
