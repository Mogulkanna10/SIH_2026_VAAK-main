import tensorflow as tf
from tensorflow.keras import layers, models


def ds_block(x, filters, stride=1):

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=True
    )(x)

    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=True
    )(x)

    x = layers.ReLU()(x)

    return x


def build_model(
    input_shape=(49, 40, 1),
    num_classes=2
):

    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )

    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=True
    )(inputs)

    x = layers.ReLU()(x)

    x = ds_block(
        x,
        24,
        stride=1
    )

    x = ds_block(
        x,
        32,
        stride=2
    )

    x = ds_block(
        x,
        48,
        stride=1
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        24,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(x)

    return models.Model(
        inputs,
        outputs,
        name="Dracarys_DS_CNN_V2"
    )


if __name__ == "__main__":

    model = build_model()

    model.summary()

    print(
        "Parameters:",
        model.count_params()
    )
