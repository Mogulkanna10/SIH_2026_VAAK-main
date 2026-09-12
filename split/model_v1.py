import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(
    input_shape=(49, 40, 1),
    num_classes=2
):
    inputs = layers.Input(
        shape=input_shape,
        name="logmel_input"
    )

    # Initial feature extraction
    x = layers.Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable block 1
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        24,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable block 2
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise-separable block 3
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        48,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Small classifier
    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Dracarys_DS_CNN"
    )


if __name__ == "__main__":

    model = build_model()

    model.summary()

    print()
    print(
        "Parameters:",
        model.count_params()
    )
