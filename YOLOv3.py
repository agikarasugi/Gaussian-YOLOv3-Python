import tensorflow as tf
import numpy as np


## Create function load_data, load_model, and custom_loss so that the training can be performed
## If there are more than 1 network, create more than one load_model function. i.e. GAN has at least 1 generator and 1 discriminator network.
## Please use the following naming convention:
## XXXX_YY_ZZ
## XXXX is the name of the layer. i.e. Convolutional Layer ==> Conv (Make the name of the layer to be 4 letters)
## YY is the number of block. i.e. 01 is the first block
## ZZ is the number of layer of the block. i.e. 02 is the second layer in the block
## i.e. Conv_03_01 is the first convolution layer in the third block
## i.e. Relu_03_01 is the first relu layer in the third block
## Decide the optimizer and train function 

anchor_file_path = './anchors.txt'
class_file_path = './voc.names'
dataset_path = './voc_train.txt'

input_data = tf.placeholder(dtype=tf.float32, name='input_data')
label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
trainable = tf.placeholder(dtype=tf.bool, name='training')


def load_data():
    # Load anchors
    with open(anchor_file_path, 'r') as anchor_file:
        anchors = anchor_file.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    anchors = anchors.reshape(3, 3, 2)
    
    # Load class names
    classnames = {}
    with open(class_file_path, 'r') as class_file:
        for ID, name in enumerate(class_file):
            classnames[ID] = name.strip('\n')

    # Load dataset/annotations
    annotations = []
    with open(dataset_path, 'r') as dataset_file:
        lines = dataset_file.readlines()
        for line in lines:
            if len(line.strip().split()[1:]) != 0:
                annotations.append(line.strip()) 

    return anchors, classnames, annotations


# Input definition

anchors, classnames, annotations = load_data()
num_class = len(classnames)
num_samples = len(annotations)
batch_size = 6
strides = np.array([8, 16, 32])


def convolution_layer(input_data, shape, trainable, downsample=False, name=None):
    with tf.variable_scope(name):
        if downsample:
            pad_h = (shape[0] - 2) // 2 + 1
            pad_w = (shape[1] - 2) // 2 + 1
            paddings = tf.constant([
                [0, 0], 
                [pad_h, pad_h], 
                [pad_w, pad_w],
                [0, 0]
            ])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'

        else:
            strides = (1, 1, 1, 1)
            padding = 'SAME'

        weight = tf.get_variable(name='weight',
            dtype=tf.float32,
            trainable=True,
            shape=shape,
            initializer=tf.random_normal_initializer(stddev=0.01))

        conv = tf.nn.conv2d(input_data, weight, padding=padding, strides=strides, name=name)

        return conv


def batch_normalization(input_data, name=None):
    bn = tf.layers.batch_normalization(
                input_data, beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=trainable,
                name=name
            )
    
    return bn


def add_bias(input_data, shape, name=None):
    bias = tf.get_variable(
                name=name,
                shape=shape[-1],
                trainable=True,
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
            )
    
    input_data = tf.nn.bias_add(input_data, bias)
    return input_data


def residual_block(input_data, input_channel, filter1, filter2, trainable, block_num):
    block_name = 'Resi_' + str(block_num)

    with tf.variable_scope(block_name):
        name = '_' + str(block_num) + '_'

        shortcut = input_data

        input_data = convolution_layer(input_data, (1, 1, input_channel, filter1), trainable, name='Conv_' + name + '1')
        input_data = batch_normalization(input_data, "Bnrm" + name + '1')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu" + name + '1')

        name = '_' + str(block_num) + '_'

        input_data = convolution_layer(input_data, (3, 3, filter1, filter2), trainable, name='Conv_' + name + '2')
        input_data = batch_normalization(input_data, "Bnrm" + name + '2')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu" + name + '2')

        input_data = input_data + shortcut
    
    return input_data


def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name):
    with tf.variable_scope(name):
        input_shape = tf.shape(input_data)
        output = tf.image.resize_nearest_neighbor(
            input_data, (input_shape[1] * 2, input_shape[2] * 2)
        )
        return output


def load_model():
    def build_network(input_data):
        # Conv 1
        input_data = convolution_layer(input_data, (3, 3, 3, 32), trainable, name='Conv_01_01')
        input_data = batch_normalization(input_data, "Bnrm_01_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_01_01")

        # Conv 2
        input_data = convolution_layer(input_data, (3, 3, 32, 64), trainable, downsample=True, name='Conv_02_01')
        input_data = batch_normalization(input_data, "Bnrm_02_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_02_01")

        # Conv 3, Conv 4
        input_data = residual_block(input_data, 64, 32, 64, trainable, '03')

        # Conv 5
        input_data = convolution_layer(input_data, (3, 3, 64, 128), trainable, downsample=True, name='Conv_04_01')
        input_data = batch_normalization(input_data, 'Bnrm_04_01')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_04_01")

        # Conv 6, Conv 7
        input_data = residual_block(input_data, 128, 64, 128, trainable, '05')
        # Conv 8, Conv 9
        input_data = residual_block(input_data, 128, 64, 128, trainable, '06')

        # Conv 10
        input_data = convolution_layer(input_data, (3, 3, 128, 256), trainable, downsample=True, name='Conv_07_01')
        input_data = batch_normalization(input_data, 'Bnrm_07_01')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_07_01")

        # Conv 11-26
        for i in range(8):
            input_data = residual_block(input_data, 256, 128, 256, trainable, "{:02d}".format(i+8))

        route_1 = input_data

        # Conv 27
        input_data = convolution_layer(input_data, (3, 3, 256, 512), trainable, downsample=True, name='Conv_16_01')
        input_data = batch_normalization(input_data, 'Bnrm_16_01')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_16_01")

        # Conv 28-43
        for i in range(8):
            input_data = residual_block(input_data, 512, 256, 512, trainable, "{:02d}".format(i+17))

        route_2 = input_data

        # Conv 44
        input_data = convolution_layer(input_data, (3, 3, 512, 1024), trainable, downsample=True, name='Conv_25_01')
        input_data = batch_normalization(input_data, 'Bnrm_25_01')
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_25_01")

        # Conv 45-52
        for i in range(4):
            input_data = residual_block(input_data, 1024, 512, 1024, trainable, "{:02d}".format(i+26))

        # Conv 53
        input_data = convolution_layer(input_data, (1, 1, 1024, 512), trainable, name='Conv_30_01')
        input_data = batch_normalization(input_data, "Bnrm_30_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_30_01")

        # Conv 54
        input_data = convolution_layer(input_data, (3, 3, 512, 1024), trainable, name='Conv_31_01')
        input_data = batch_normalization(input_data, "Bnrm_31_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_31_01")

        # Conv 55
        input_data = convolution_layer(input_data, (1, 1, 1024, 512), trainable, name='Conv_32_01')
        input_data = batch_normalization(input_data, "Bnrm_32_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_32_01")

        # Conv 56
        input_data = convolution_layer(input_data, (3, 3, 512, 1024), trainable, name='Conv_33_01')
        input_data = batch_normalization(input_data, "Bnrm_33_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_33_01")

        # Conv 57
        input_data = convolution_layer(input_data, (1, 1, 1024, 512), trainable, name='Conv_34_01')
        input_data = batch_normalization(input_data, "Bnrm_34_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_34_01")

        ### Large-sized object outputs ###
        lobj_branch = convolution_layer(input_data, (3, 3, 512, 1024), trainable, name='Conv_35_01')
        lobj_branch = batch_normalization(lobj_branch, "Bnrm_35_01")
        lobj_branch = tf.nn.leaky_relu(lobj_branch, alpha=0.1, name="Relu_35_01")

        lbbox = convolution_layer(lobj_branch, (1, 1, 1024, 3*(num_class + 5)), trainable, name='Conv_36_01')
        lbbox = add_bias(lbbox, (1, 1, 1024, 3*(num_class + 5)), 'Bias_36_01')
        ### ###

        # Conv 58
        input_data = convolution_layer(input_data, (1, 1, 512, 256), trainable, name='Conv_37_01')
        input_data = batch_normalization(input_data, "Bnrm_37_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_37_01")
        input_data = upsample(input_data, 'Upsm_37_01')

        input_data = route('route_1', route_2, input_data)

        # Conv 59
        input_data = convolution_layer(input_data, (1, 1, 768, 256), trainable, name='Conv_38_01')
        input_data = batch_normalization(input_data, "Bnrm_38_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_38_01")

        # Conv 60
        input_data = convolution_layer(input_data, (3, 3, 256, 512), trainable, name='Conv_39_01')
        input_data = batch_normalization(input_data, "Bnrm_39_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_39_01")

        # Conv 61
        input_data = convolution_layer(input_data, (1, 1, 512, 256), trainable, name='Conv_40_01')
        input_data = batch_normalization(input_data, "Bnrm_40_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_40_01")

        # Conv 62
        input_data = convolution_layer(input_data, (3, 3, 256, 512), trainable, name='Conv_41_01')
        input_data = batch_normalization(input_data, "Bnrm_41_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_41_01")

        # Conv 63
        input_data = convolution_layer(input_data, (1, 1, 512, 256), trainable, name='Conv_42_01')
        input_data = batch_normalization(input_data, "Bnrm_42_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_42_01")

        ### Medium-sized object outputs ###
        mobj_branch = convolution_layer(input_data, (3, 3, 256, 512), trainable, name='Conv_43_01')
        mobj_branch = batch_normalization(mobj_branch, "Bnrm_43_01")
        mobj_branch = tf.nn.leaky_relu(mobj_branch, alpha=0.1, name="Relu_43_01")

        mbbox = convolution_layer(mobj_branch, (1, 1, 512, 3*(num_class + 5)), trainable, name='Conv_44_01')
        mbbox = add_bias(mbbox, (1, 1, 512, 3*(num_class + 5)), 'Bias_44_01')
        ### ###

        # Conv 64
        input_data = convolution_layer(input_data, (1, 1, 256, 128), trainable, name='Conv_45_01')
        input_data = batch_normalization(input_data, "Bnrm_45_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_45_01")
        input_data = upsample(input_data, 'Upsm_45_01')

        input_data = route('route_2', route_1, input_data)

        # Conv 65
        input_data = convolution_layer(input_data, (1, 1, 384, 128), trainable, name='Conv_46_01')
        input_data = batch_normalization(input_data, "Bnrm_46_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_46_01")

        # Conv 66
        input_data = convolution_layer(input_data, (3, 3, 128, 256), trainable, name='Conv_47_01')
        input_data = batch_normalization(input_data, "Bnrm_47_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_47_01")

        # Conv 67
        input_data = convolution_layer(input_data, (1, 1, 256, 128), trainable, name='Conv_48_01')
        input_data = batch_normalization(input_data, "Bnrm_48_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_48_01")

        # Conv 68
        input_data = convolution_layer(input_data, (3, 3, 128, 256), trainable, name='Conv_49_01')
        input_data = batch_normalization(input_data, "Bnrm_49_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_49_01")

        # Conv 69
        input_data = convolution_layer(input_data, (1, 1, 256, 128), trainable, name='Conv_50_01')
        input_data = batch_normalization(input_data, "Bnrm_50_01")
        input_data = tf.nn.leaky_relu(input_data, alpha=0.1, name="Relu_50_01")

        ### Small-sized object outputs ###
        sobj_branch = convolution_layer(input_data, (3, 3, 128, 256), trainable, name='Conv_51_01')
        sobj_branch = batch_normalization(sobj_branch, "Bnrm_51_01")
        sobj_branch = tf.nn.leaky_relu(sobj_branch, alpha=0.1, name="Relu_51_01")

        sbbox = convolution_layer(sobj_branch, (1, 1, 256, 3*(num_class + 5)), trainable, name='Conv_52_01')
        sbbox = add_bias(sbbox, (1, 1, 512, 3*(num_class + 5)), 'Bias_52_01')
        ### ###

        return lbbox, mbbox, sbbox
    return build_network


def decode_bbox(output, anchors, stride):
    conv_shape = tf.shape(output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = tf.reshape(output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def custom_loss(outputs, dec_bboxes, sbbox_label, true_sbbox, mbbox_label, true_mbbox, lbbox_label, true_lbbox):
    
    def focal(target, actual, a=1, g=2):
        return a * tf.pow(tf.abs(target - actual), g)

    def bbox_giou(boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou
    
    def bbox_iou(boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(output, prediction, label, bboxes, anchors, stride):
        conv_shape = tf.shape(output)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        output = tf.reshape(
            output, (batch_size, output_size, output_size, 3, 5 + num_class)
        )

        conv_raw_conf = output[:, :, :, :, 4:5]
        conv_raw_prob = output[:, :, :, :, 5:]

        pred_xywh = prediction[:, :, :, :, 0:4]
        pred_conf = prediction[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < 0.5, tf.float32)

        conf_focal = focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss():
        loss_sbbox = loss_layer(
            outputs[2], dec_bboxes[2], 
            sbbox_label, true_sbbox,
            anchors[0], strides[0]
        )

        loss_mbbox = loss_layer(
            outputs[1], dec_bboxes[1], 
            mbbox_label, true_mbbox,
            anchors[1], strides[1]
        )

        loss_lbbox = loss_layer(
            outputs[0], dec_bboxes[0], 
            lbbox_label, true_lbbox,
            anchors[2], strides[2]
        )

        giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        return giou_loss + conf_loss + prob_loss

    return compute_loss()


# Model definition

cnn_model = load_model()

# Output definition

input_images = input_data

output = cnn_model(input_images)

decoded_lbbox = decode_bbox(output[0], anchors[2], strides[2])
decoded_mbbox = decode_bbox(output[1], anchors[1], strides[1])
decoded_sbbox = decode_bbox(output[2], anchors[0], strides[0])

# Loss definition

loss = custom_loss(
    output, 
    (decoded_lbbox, decoded_mbbox, decoded_sbbox),
    label_sbbox, true_sbboxes,
    label_mbbox, true_lbboxes,
    label_lbbox, true_lbboxes
)

# Optimizer

optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)
