import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Attention, \
    UpSampling2D, Conv2DTranspose
from tensorflow.keras import Input
from tensorflow.keras.models import Model


def attnet(in_height, in_width):
    inp = Input(shape=(in_height, in_width, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', name='attn_block1_conv1')(inp)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', name='attn_block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='attn_block1_pool')(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', name='attn_block2_conv1')(pool1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', name='attn_block2_conv2')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='attn_block2_pool')(conv4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', name='attn_block3_conv1')(pool2)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', name='attn_block3_conv2')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='attn_block3_pool')(conv6)
    conv7 = Conv2D(512, 3, activation='relu', padding='same', name='attn_block4_conv1')(pool3)
    conv8 = Conv2D(512, 3, activation='relu', padding='same', name='attn_block4_conv2')(conv7)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='attn_block4_pool')(conv8)
    conv9 = Conv2D(1024, 3, activation='relu', padding='same', name='attn_block5_conv1')(pool4)
    conv10 = Conv2D(1024, 3, activation='relu', padding='same', name='attn_block5_conv2')(conv9)
    up1 = Conv2D(512, 2, activation='relu', padding='same', name='attn_block5_upsample')(UpSampling2D(size=(2, 2))(conv10))
    merge1 = Concatenate(axis=3, name='attn_block4_merge')([conv8, up1])
    conv11 = Conv2D(512, 3, activation='relu', padding='same', name='attn_block4_conv3')(merge1)
    conv12 = Conv2D(512, 3, activation='relu', padding='same', name='attn_block4_conv4')(conv11)
    up2 = Conv2D(256, 2, activation='relu', padding='same', name='attn_block4_upsample')(UpSampling2D(size=(2, 2))(conv12))
    merge2 = Concatenate(axis=3, name='attn_block3_merge')([conv6, up2])
    conv13 = Conv2D(256, 3, activation='relu', padding='same', name='attn_block3_conv3')(merge2)
    conv14 = Conv2D(256, 3, activation='relu', padding='same', name='attn_block3_conv4')(conv13)
    up3 = Conv2D(128, 2, activation='relu', padding='same', name='attn_block3_upsample')(UpSampling2D(size=(2, 2))(conv14))
    merge3 = Concatenate(axis=3, name='attn_block2_merge')([conv4, up3])
    conv15 = Conv2D(128, 3, activation='relu', padding='same', name='attn_block2_conv3')(merge3)
    conv16 = Conv2D(128, 3, activation='relu', padding='same', name='attn_block2_conv4')(conv15)
    up4 = Conv2D(64, 2, activation='relu', padding='same', name='attn_block2_upsample')(UpSampling2D(size=(2, 2))(conv16))
    merge4 = Concatenate(axis=3, name='attn_block1_merge')([conv2, up4])
    conv17 = Conv2D(64, 3, activation='relu', padding='same', name='attn_block1_conv3')(merge4)
    conv18 = Conv2D(64, 3, activation='relu', padding='same', name='attn_block1_conv4')(conv17)
    conv19 = Conv2D(1, 1, activation='sigmoid', name='attn_out')(conv18)
    return Model(inputs=inp, outputs=conv19)

def EM(input, kernal_size, channel, name):
    conv_1 = Conv2D(channel, (3, 3), activation='relu', padding='same', name=name + '_conv1')(input)
    conv_2 = Conv2D(channel, (kernal_size, kernal_size), activation='relu', padding='valid',
                    name=name + "_conv2")(conv_1)
    conv_3 = Conv2D(channel * 2, (kernal_size, kernal_size), activation='relu', padding='valid',
                    name=name + "_conv3")(conv_2)
    conv_4 = Conv2D(channel * 4, (kernal_size, kernal_size), activation='relu', padding='valid',
                    name=name + "_conv4")(conv_3)
    conv_5 = Conv2DTranspose(channel * 2, (kernal_size, kernal_size), activation='relu',
                             padding='valid', name=name + "_deconv1")(conv_4)
    conv_6 = Conv2DTranspose(channel, (kernal_size, kernal_size), activation='relu',
                             padding='valid', name=name + "_deconv2")(conv_5)
    res = Conv2DTranspose(3, (kernal_size, kernal_size), activation='relu', padding='valid',
                          name=name + "_deconv3")(conv_6)
    return res

def enhance_net(in_height, in_width):
    inp = Input(shape=(in_height, in_width, 4))
    fem1 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv1")(inp)
    em1 = EM(fem1, 5, 8, "em1")
    fem2 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv2")(fem1)
    em2 = EM(fem2, 5, 8, "em2")
    fem3 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv3")(fem2)
    em3 = EM(fem3, 5, 8, "em3")
    fem4 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv4")(fem3)
    em4 = EM(fem4, 5, 8, "em4")
    fem5 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv5")(fem4)
    em5 = EM(fem5, 5, 8, "em5")
    fem6 = Conv2D(32, (3, 3), activation='relu', padding='same', name="fem_conv6")(fem5)
    em6 = EM(fem6, 5, 8, "em6")
    merge = Concatenate(axis=3, name="fem_merge")([em1, em2, em3, em4, em5, em6])
    output = Conv2D(3, (1, 1), activation='sigmoid', padding='same', name="output")(merge)
    return Model(inputs=inp, outputs=output)
