import tensorflow as tf


def get_augmentation(flip_up_down=False, flip_left_right=False, height=False, noise=False, defects=False, blur=False, dual=False):
    def gaussian_kernel():
        d = tf.distributions.Normal(0.0, tf.random_uniform((), 0, 0.5))
        size = 2
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def augmentation(image, label):
        if flip_up_down:
            image = tf.image.random_flip_up_down(image)
        if flip_left_right:
            image = tf.image.random_flip_left_right(image)
        if height:
            delta_max = 1 - tf.reduce_max(image, axis=(1, 2))
            delta_min = -tf.reduce_min(image + tf.to_float(tf.equal(image, 0.0)), axis=(1, 2))
            delta = tf.random_uniform(tf.shape(delta_max), delta_min, delta_max)
            delta = tf.expand_dims(tf.expand_dims(delta, axis=2), axis=3)
            mask = tf.to_float(tf.greater(image, 0.0))  # (0 where black, else 1)
            image = tf.clip_by_value(image + delta * mask, 0.0, 1.0)
        if blur:
            gauss_kernel = gaussian_kernel()
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            image = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
        if noise:
            image += tf.random_normal(tf.shape(image), mean=0.0, stddev=tf.random_uniform((), 0, 0.005))
        if defects:
            ones = tf.ones([tf.shape(image)[0], 8, 8, 1])
            mask = tf.nn.dropout(ones, 0.95)
            mask = tf.image.resize_images(mask, (32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            image = image * mask
        return image, label
    
    def dual_augmentation(depth_image, raw_image, label):
        if flip_up_down:
            mirror_cond = tf.random_uniform([], 0, 1.0) < 0.5
            depth_image = tf.cond(mirror_cond, lambda: tf.image.flip_up_down(depth_image), lambda: depth_image)
            raw_image = tf.cond(mirror_cond, lambda: tf.image.flip_up_down(raw_image), lambda: raw_image)
        if flip_left_right:
            mirror_cond = tf.random_uniform([], 0, 1.0) < 0.5
            depth_image = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(depth_image), lambda: depth_image)
            raw_image = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(raw_image), lambda: raw_image)
        return depth_image, raw_image, label
    if dual:
        return dual_augmentation
    return augmentation


# Index index is the index, where the index in the output is given.
def single_class_split(y_true, y_pred, index_index=1):
    value_true = y_true[:, 0]
    index = tf.to_int32(y_true[:, index_index])
    indices_gripper_class = tf.stack([tf.range(tf.shape(index)[0]), index], axis=1)
    value_pred = tf.gather_nd(y_pred, indices_gripper_class)
    return value_true, value_pred


def tf_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.round(y_pred), y_true)))


def tf_precision(y_true, y_pred):
    return tf.metrics.precision(y_true, tf.round(y_pred))[1]


def tf_recall(y_true, y_pred):
    return tf.metrics.recall(y_true, tf.round(y_pred))[1]


def tf_f1(precision, recall, beta=0.5):
    return (1. + beta**2) * (precision * recall) / (beta**2 * precision + recall)
