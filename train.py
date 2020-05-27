import Network
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg19 import VGG19
import os
import argparse


def get_ground_truth_path(file_path):
    path_mid = tf.strings.split(file_path, os.path.sep)
    return tf.strings.join([bright_path, path_mid[-1]], os.path.sep)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [height, width])


def process_path(file_path):
    gt_path = get_ground_truth_path(file_path)
    gt = tf.io.read_file(gt_path)
    gt = decode_img(gt)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, gt


def data_gen():
    list_ds = tf.data.Dataset.list_files(os.path.join(dark_path, '*.jpg')).shuffle(22656)
    return list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


def contrast(image_batch):
    im_gray = tf.image.rgb_to_grayscale(image_batch)
    return tf.math.reduce_std(im_gray, [1, 2, 3])


def bright_loss(y_true, y_pred):
    return tf.losses.mae(y_true, y_pred)


def structural_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    return tf.reduce_mean(ssim)


def regional_loss(y_true, y_pred, att_map):
    true = y_true * att_map
    pred = y_pred * att_map
    return tf.losses.mae(true, pred) + 1 - structural_loss(true, pred)


def contrast_loss(y_true, y_pred):
    true_cont = contrast(y_true)
    pred_cont = contrast(y_pred)
    return tf.losses.mae(true_cont, pred_cont)


vgg_model = VGG19(include_top=False, weights='imagenet')
vgg = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv4').output)


def perceptual_loss(y_true, y_pred):
    vgg_pred = tf.reshape(vgg(2 * y_pred - 1), [-1, 256, 256, 16])
    vgg_true = tf.reshape(vgg(2 * y_true - 1), [-1, 256, 256, 16])
    return tf.losses.mean_absolute_error(vgg_true, vgg_pred)


def light_loss(y_true, y_pred, att_map):
    b_loss = bright_loss(y_true, y_pred)
    c_loss = contrast_loss(y_true, y_pred)
    s_loss = 1 - structural_loss(y_true, y_pred)
    r_loss = 5 * regional_loss(y_true, y_pred, att_map)
    p_loss = 0.35 * perceptual_loss(y_true, y_pred)
    return b_loss + c_loss + s_loss + r_loss + p_loss


optimizer = Adam(learning_rate=0.0002, epsilon=1e-8)
train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def att_module_train_step(model, dark_images, actual_images):
    with tf.GradientTape() as tape:
        bright_max = tf.reduce_max(actual_images, axis=-1)
        att_map = tf.abs((bright_max - tf.reduce_max(dark_images, axis=-1))) / (bright_max + 1e-8)[..., tf.newaxis]
        att_map = tf.clip_by_value(att_map, 0, 1)
        predictions = model(dark_images)
        loss = tf.losses.mae(att_map, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


def get_train_func_for_enhancement(att_net):
    @tf.function
    def train_step(model, dark_images, actual_images):
        with tf.GradientTape() as tape:
            att_map = att_net(dark_images)
            pr_inp = tf.concat([dark_images, att_map], -1)
            predictions = model(pr_inp)
            loss = light_loss(actual_images, predictions, att_map)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
    return train_step


def train_model(model, epochs, model_path, step_func):
    for epoch in range(epochs):
        gen = data_gen()
        steps = 0
        for dark, bright in gen:
            step_func(model, dark, bright)
            steps += 1
            if steps % 354 == 0:
                model.save_weights(model_path)
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch+1, train_loss.result()))
                print(" -- ", flush=True)

        # Reset the metrics for the next epoch
        train_loss.reset_states()


def train_attention_module(epochs, model_path, height, width):
    model = Network.attnet(height, width)
    train_model(model, epochs, model_path, att_module_train_step)


def train_enhacment_module(epochs, enhancment_module_path, attention_module_path, height, width):
    att_net = load_model(attention_module_path)
    model = Network.enhance_net(height, width)
    step_func = get_train_func_for_enhancement(att_net)
    train_model(model, epochs, enhancment_module_path, step_func)


def train_complete_net(epochs, enhancement_module_path, attention_module_path, height, width):
    att_module = Network.attnet(height, width)
    enhancement_module = Network.enhance_net(height, width)
    enhance_train_step = get_train_func_for_enhancement(att_module)
    for epoch in range(epochs):
        gen = data_gen()
        steps = 0
        for dark, bright in gen:
            att_module_train_step(att_module, dark, bright)
            enhance_train_step(enhancement_module, dark, bright)
            steps += 1
            if steps % 354 == 0:
                att_module.save_weights(attention_module_path)
                enhancement_module.save_weights(enhancement_module_path)
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, train_loss.result()))
                print(" -- ", flush=True)

        # Reset the metrics for the next epoch
        train_loss.reset_states()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark_path", type=str, required=True, help="dark images path")
    parser.add_argument("--bright_path", type=str, required=True, help="bright images path")
    parser.add_argument("--attention_module_path", type=str, default="attnet.h5", help="file to contain the attention module")
    parser.add_argument("--enhancement_module_path", type=str, default="light.h5", help="file to contain the enhancement module")
    parser.add_argument("--train_mode", type=str, choices=["attention_only", "enhancement_only", "combined"], help="part of the network to train", required=True)
    parser.add_argument("--height", type=int, default=256, help="image height for resizing")
    parser.add_argument("--width", type=int, default=256, help="image width for resizing")
    parser.add_argument("epochs", type=int, default=10, help="number of training epochs")
    arg = parser.parse_args()
    height = arg.height
    width = arg.width
    dark_path = arg.dark_path
    bright_path = arg.bright_path
    att_module_path = arg.attention_module_path
    enhance_path = arg.enhancement_module_path
    mode = arg.train_mode
    epochs = arg.epochs
    if mode == "attention_only":
        train_attention_module(epochs, att_module_path, height, width)
    elif mode == "enhancement_only":
        train_enhacment_module(epochs, enhance_path, att_module_path, height, width)
    else:
        train_complete_net(epochs, enhance_path, att_module_path, height, width)
