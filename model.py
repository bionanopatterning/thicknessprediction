import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import datetime
import colorcet as cc
import time
from IPython import display
import datetime

### ### ### ### ### ### ### ### SETTINGS ### ### ### ### ### ### ### ###


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_progress_folder = "C:/Users/mgflast/Desktop/rgb_train_data/" + timestamp + "_training_progress/"

TRAIN_SIZE = 500
BATCH_SIZE = 8
IMG_WIDTH = 256
IMG_HEIGHT = 256
HEIGHT_RESCALE = 1 / 1000.0  # 1 nm -> this value (to make it fit in range 0.0 - 1.0)
OUTPUT_CHANNELS = 1

# Going along with Tensorflow pix2pix example.
# https://www.tensorflow.org/tutorials/generative/pix2pix

cet_cmap_error = cc.mpl_cl("cet_cmap", cc.CET_D4)
cet_cmap_main = cc.mpl_cl("cet_cma2", cc.CET_R4)


def load(idx, folder):
    train_x = np.asarray(Image.open(folder + f"00{idx}_reflection.png")) / 127.5 - 1.0
    heightmap = np.asarray(Image.open(folder + f"00{idx}_thickness.tiff")) * HEIGHT_RESCALE

    train_y = np.zeros((256, 256, 1))
    train_y[:, :, 0] = heightmap

    train_x = train_x[:, :, :]
    train_x = tf.cast(train_x, tf.float32)
    train_y = tf.cast(train_y, tf.float32)
    train_y = train_y[:, :, :]

    return train_x, train_y


def load_dataset(folder):
    heightmaps = list()
    rgbimgs = list()
    for i in range(TRAIN_SIZE):
        rgb, hmap = load(i, folder)
        heightmaps.append(hmap)
        rgbimgs.append(rgb)
    train_dataset = tf.data.Dataset.from_tensor_slices((rgbimgs, heightmaps))
    return train_dataset


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                      use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                               use_bias=False))
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
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')

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
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
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
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, generator=generator,
                                 discriminator=discriminator)

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
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


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.title("Input RGB image")
    plt.imshow(test_input[0] * 0.5 + 0.5)
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.title("Ground truth")
    plt.imshow(tar[0] / HEIGHT_RESCALE, cmap=cet_cmap_main, vmin=0.0, vmax=1000.0)
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.title("Prediction")
    plt.imshow(prediction[0] / HEIGHT_RESCALE, cmap=cet_cmap_main, vmin=0.0, vmax=1000.0)
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.title("Error")
    plt.imshow((tar[0] - prediction[0]) / HEIGHT_RESCALE, cmap=cet_cmap_error, vmin=-100.0, vmax=100.0)
    plt.axis("off")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(train_progress_folder + timestamp + ".png")
    plt.close()


def fit(train_ds, test_ds, steps):
    start = time.time()
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    example_input, example_target = next(iter(train_ds.take(int(np.random.uniform(0, TRAIN_SIZE)))))
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()

            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step) % 10 == 0:
            print('.', end='', flush=True)
        if (step) % 250 == 0:
            generate_images(generator, example_input, example_target)
        # Save (checkpoint) the model every 1k steps
        if (step + 1) % 1000 == 0:
            manager.save()


def train_model(train_data_folder, epochs=100):
    train_dataset = load_dataset(train_data_folder)
    train_dataset = train_dataset.shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
    fit(train_dataset, None, steps=TRAIN_SIZE * epochs)


def load_model(model_path):
    global checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(model_path))


def test_model(rgb_image_path, ground_truth=None, savepath=None):
    rgb_image = np.asarray(Image.open(rgb_image_path)) / 127.5 - 1.0

    rgb_image = rgb_image[None, :, :, :]
    rgb_image = tf.cast(rgb_image, tf.float32)

    prediction = generator(rgb_image, training=True)

    plt.figure("Testing model")
    plt.subplot(2, 2, 1)
    plt.title("Input image")
    plt.imshow(rgb_image[0] * 0.5 + 0.5)

    plt.subplot(2, 2, 2)
    plt.title("Output: height map")
    plt.imshow(prediction[0, :, :, 0] / HEIGHT_RESCALE, cmap="jet")
    if savepath:
        Image.fromarray(np.asarray(prediction[0, :, :, 0]) / HEIGHT_RESCALE).save(savepath + "prediction.tiff")

    if ground_truth:
        gt = np.asarray(Image.open(ground_truth))
        plt.subplot(2, 2, 3)
        plt.title("Ground truth")
        plt.imshow(gt, cmap="jet")

        plt.subplot(2, 2, 4)
        errormap = ((prediction[0, :, :, 0] / HEIGHT_RESCALE) - gt)
        plt.title("Prediction error\n (nm)")
        plt.imshow(errormap, cmap="jet")

    plt.show()


def rgb_to_height_pathinput(rgb_image_path):
    rgb_image = np.asarray(Image.open(rgb_image_path)) / 127.5 - 1.0
    rgb_image = rgb_image[None, :, :, :]
    rgb_image = tf.cast(rgb_image, tf.float32)

    prediction = generator(rgb_image, training=True)
    imgout = np.asarray(prediction[0, :, :, 0] / HEIGHT_RESCALE)
    return imgout, rgb_image[0, :, :, :] * 0.5 + 0.5


def rgb_to_height_imginput(rgb_image):
    rgb_image = rgb_image / 127.5 - 1.0
    rgb_image = rgb_image[None, :, :, :]
    rgb_image = tf.cast(rgb_image, tf.float32)

    prediction = generator(rgb_image, training=True)
    imgout = np.asarray(prediction[0, :, :, 0] / HEIGHT_RESCALE)
    return imgout, rgb_image[0, :, :, :] * 0.5 + 0.5


def train_from_checkpoint(training_dataset, epochs=5, checkpoint=None):
    def load_model(model_path):
        global checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(model_path))

    load_model(checkpoint)
    train_model(training_dataset, epochs)


def clear_memory():
    tf.keras.backend.clear_session()