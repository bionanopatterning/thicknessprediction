import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import datetime
import colorcet as cc

train_data_folder = "C:/Users/mgflast/Desktop/rgb_train_data/test_set_3000/"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_progress_folder = "C:/Users/mgflast/Desktop/rgb_train_data/" + timestamp + "_training_progress/"

TRAIN_SIZE = 490
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
HEIGHT_RESCALE = 1 / 1000.0 # 1 nm -> this value (to make it fit in range 0.0 - 1.0)
OUTPUT_CHANNELS = 3


# Going along with Tensorflow pix2pix example.
# https://www.tensorflow.org/tutorials/generative/pix2pix

cet_cmap_error = cc.mpl_cl("cet_cmap", cc.CET_D4)
cet_cmap_main = cc.mpl_cl("cet_cma2", cc.CET_C2s)

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis = 0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, 1 , IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]

def load(idx):
    train_x = np.asarray(Image.open(train_data_folder + f"00{idx}_rgb.png")) / 127.5 - 1.0
    heightmap = np.asarray(Image.open(train_data_folder + f"00{idx}_height.tiff")) * HEIGHT_RESCALE
    train_y = np.zeros_like(train_x)
    train_y[:, :, 0] = heightmap
    train_y[:, :, 1] = heightmap
    train_y[:, :, 2] = heightmap

    train_x = train_x[None, :, :, :]
    train_x = tf.cast(train_x, tf.float32)
    train_y = tf.cast(train_y, tf.float32)
    train_y = train_y[None, :, :, :]

    # unlike in Tensorflow example, no jitter applied to the input. (avoids interpolation artefacts in the composite img).

    if np.random.uniform() > 0.5:
        train_x = tf.image.flip_left_right(train_x)
        train_y = tf.image.flip_left_right(train_y)
    if np.random.uniform() > 0.5:
        train_x = tf.image.flip_up_down(train_x)
        train_y = tf.image.flip_up_down(train_y)

    return train_x, train_y


def plot(tx, ty):
    plt.subplot(1,2,1)
    plt.title("RGB Reflection")
    plt.imshow(tx * 0.5 + 0.5)
    plt.subplot(1,2,2)
    plt.title("Height map (ground truth)")
    plt.imshow(ty, cmap = "jet")
    plt.show()


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',kernel_initializer = initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip, in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer = initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

generator = Generator()
discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = timestamp + "_checkpoints/"

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

def show_sample(model, test_input, target, epoch, stepcount):
    prediction = model(test_input, training = True)

    plt.figure("Training output for debug", figsize = (15, 15))

    plt.subplot(2, 2, 1)
    plt.title("Input:\nRGB reflected light")
    plt.imshow(test_input[0] * 0.5 + 0.5)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Ground truth:\nheight map")
    vmin = np.amin(target[0, :, :, 0]) / HEIGHT_RESCALE
    vmax = np.amax(target[0, :, :, 0]) / HEIGHT_RESCALE
    plt.imshow(target[0, :, :, 0] / HEIGHT_RESCALE, vmin = vmin, vmax = vmax, cmap = cet_cmap_main)

    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Prediction:\ngenerated height map")
    plt.imshow(prediction[0, :, :, 0] / HEIGHT_RESCALE, vmin = vmin, vmax = vmax, cmap = cet_cmap_main)
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("Difference:\ntruth - prediction \n(range -100 - 100.0)")
    plt.imshow((target[0, :, :, 0] - prediction[0, :, :, 0]) / HEIGHT_RESCALE, vmin = -100, vmax = 100, cmap = cet_cmap_error)
    plt.axis("off")
    try:
        os.makedirs(train_progress_folder)
    except Exception as e:
        pass
    plt.savefig(train_progress_folder + timestamp + "_progress_" + f"epoch {epoch} step {stepcount}.png")
    print(train_progress_folder + timestamp + "_progress_" + f"epoch {epoch} step {stepcount}.png")

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training = True)
        disc_real_output = discriminator([input_image, target], training = True)
        disc_generated_output = discriminator([input_image, gen_output], training = True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 100)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 100)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 100)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 100)

def fit(epochs):
    total_stepcount = 0
    for e in range(epochs):
        print(f"Starting epoch {e}")
        idx = np.asarray(range(0, TRAIN_SIZE))
        np.random.shuffle(idx)
        epoch_stepcount = 0
        for i in idx:
            print(f"\t processing image {epoch_stepcount}")
            if epoch_stepcount % 50 == 0:
                example_input, example_target = load(int(np.random.uniform(low=0, high=TRAIN_SIZE)))
                show_sample(generator, example_input, example_target, e, total_stepcount)
            input_image, target = load(i)
            train_step(input_image, target, total_stepcount)
            del input_image, target
            total_stepcount += 1
            epoch_stepcount += 1

        print(f"Saving checkpoint after epoch {e}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep = 3)
        manager.save()

def train_model(training_dataset, buffersize, epochs = 100):
    global train_data_folder, TRAIN_SIZE
    train_data_folder = training_dataset
    TRAIN_SIZE = buffersize
    fit(epochs)

def load_model(model_path):
    global checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(model_path))

def test_model(rgb_image_path, ground_truth = None, savepath = None):
    rgb_image = np.asarray(Image.open(rgb_image_path)) / 127.5 - 1.0

    rgb_image = rgb_image[None, :, :, :]
    rgb_image = tf.cast(rgb_image, tf.float32)

    prediction = generator(rgb_image, training=True)

    plt.figure("Testing model")
    plt.subplot(2,2,1)
    plt.title("Input image")
    plt.imshow(rgb_image[0] * 0.5 + 0.5)

    plt.subplot(2,2,2)
    plt.title("Output: height map")
    plt.imshow(prediction[0, :, :, 0] / HEIGHT_RESCALE, cmap = "jet")
    if savepath:
        Image.fromarray(np.asarray(prediction[0, :, :, 0]) / HEIGHT_RESCALE).save(savepath + "prediction.tiff")

    if ground_truth:
        gt = np.asarray(Image.open(ground_truth))
        plt.subplot(2,2,3)
        plt.title("Ground trutch")
        plt.imshow(gt, cmap = "jet")

        plt.subplot(2,2,4)
        errormap = ((prediction[0, :, :, 0] / HEIGHT_RESCALE) - gt)
        plt.title("Prediction error\n (nm)")
        plt.imshow(errormap, cmap = "jet")

    plt.show()

def rgb_to_height(rgb_image_path):
    rgb_image = np.asarray(Image.open(rgb_image_path)) / 127.5 - 1.0

    rgb_image = rgb_image[None, :, :, :]
    rgb_image = tf.cast(rgb_image, tf.float32)

    prediction = generator(rgb_image, training=True)
    imgout = np.asarray(prediction[0, :, :, 0] / HEIGHT_RESCALE)
    return imgout

def train_from_checkpoint(training_dataset, buffersize, epochs = 5, checkpoint = None):
    def load_model(model_path):
        global checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(model_path))

    load_model(checkpoint)
    train_model(training_dataset, buffersize, epochs)

def clear_memory():
    tf.keras.backend.clear_session()