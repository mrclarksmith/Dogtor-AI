# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:21:48 2021

@author: serverbob
"""
import tensorflow as tf
import tensorflow_addons as tfa
import sys
from sklearn.model_selection import train_test_split 
import os
import numpy as np
import yaml
import abc



# def get_padding(kernel_size, dilation=1):
#     return int((kernel_size*dilation - dilation)/2)
import soundfile as sf
from base_trainer import GanBasedTrainer 
from weight_norm import WeightNormalization
from preprocess_tf import PreprocessData
from group_conv  import GroupConv1D

# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)




def return_strategy():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(physical_devices) == 1:
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        return tf.distribute.MirroredStrategy()
STRATEGY = return_strategy()

config_path =  "D:\python2\woof_friend\TensorFlowTTS\examples\hifigan\conf\hifigan.v1.yaml"

# load and save config
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)
config["version"] = "test"


LRELU_SLOPE = 0.1

def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.glorot_normal` with the given seed.
    Args:
        initializer_seed: int, initializer seed.
    Returns:
        GlorotNormal initializer with seed = `initializer_seed`.
    """
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)


class TFReflectionPad1d(tf.keras.layers.Layer): #adding padding
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0]],
            self.padding_type,
        )

sys.path.append(".")


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        """Initialize TFConvTranspose1d( module.
        Args:
            filters (int): Number of filters.
            kernel_size (int): kernel size.
            strides (int): Stride width.
            padding (str): Padding type ("same" or "valid").
        """
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed),
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x




class TFHifiResBlock(tf.keras.layers.Layer):
    """Tensorflow Hifigan resblock 1 module."""

    def __init__(
        self,
        kernel_size,
        filters,
        dilation_rate,
        use_bias,
        nonlinear_activation,
        nonlinear_activation_params,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        """Initialize TFHifiResBlock module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (list): List dilation rate.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            is_weight_norm (bool): Whether to use weight norm or not.
        """
        super().__init__(**kwargs)
        self.blocks_1 = []
        self.blocks_2 = []

        for i in range(len(dilation_rate)):
            self.blocks_1.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate[i]),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=dilation_rate[i],
                        use_bias=use_bias,
                    ),
                ]
            )
            self.blocks_2.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * 1),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=1,
                        use_bias=use_bias,
                    ),
                ]
            )

        self.activation = getattr(tf.keras.layers, nonlinear_activation)(
            **nonlinear_activation_params
        )

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks_1)
            self._apply_weightnorm(self.blocks_2)

    def call(self, x, training=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        for c1, c2 in zip(self.blocks_1, self.blocks_2):
            xt = self.activation(x)
            for c in c1:
                xt = c(xt)
            xt = self.activation(xt)
            for c in c2:
                xt = c(xt)
            x = xt + x
        return x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


class TFMultiHifiResBlock(tf.keras.layers.Layer):
    """Tensorflow Multi Hifigan resblock 1 module."""

    def __init__(self, list_resblock, **kwargs):
        super().__init__(**kwargs)
        self.list_resblock = list_resblock

    def call(self, x, training=False):
        xs = None
        for resblock in self.list_resblock:
            if xs is None:
                xs = resblock(x, training=training)
            else:
                xs += resblock(x, training=training)
        return xs / len(self.list_resblock)







class TFHifiGANGenerator(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # check hyper parameter is valid or not
        assert (
            config.stacks
            == len(config.stack_kernel_size)
            == len(config.stack_dilation_rate)
        )

        # add initial layer
        layers = []
        layers += [
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="first_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.filters,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
            ),
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(
                    **config.nonlinear_activation_params
                ),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding="same",
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name="conv_transpose_._{}".format(i),
                ),
            ]

            # add residual stack layer
            layers += [
                TFMultiHifiResBlock(
                    list_resblock=[
                        TFHifiResBlock(
                            kernel_size=config.stack_kernel_size[j],
                            filters=config.filters // (2 ** (i + 1)),
                            dilation_rate=config.stack_dilation_rate[j],
                            use_bias=config.use_bias,
                            nonlinear_activation=config.nonlinear_activation,
                            nonlinear_activation_params=config.nonlinear_activation_params,
                            is_weight_norm=config.is_weight_norm,
                            initializer_seed=config.initializer_seed,
                            name="hifigan_resblock_._{}".format(j),
                        )
                        for j in range(config.stacks)
                    ],
                    name="multi_hifigan_resblock_._{}".format(i),
                )
            ]
        # add final layer
        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(
                **config.nonlinear_activation_params
            ),
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="last_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.out_channels,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
                dtype=tf.float32,
            ),
        ]
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.hifigan = tf.keras.models.Sequential(layers)

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, T, channels)
        Returns:
            Tensor: Output tensor (B, T ** prod(upsample_scales), out_channels)
        """
        return self.inference(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference(self, mels):
        return self.hifigan(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference_tflite(self, mels):
        return self.hifigan(mels)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
        self(fake_mels)




class TFHifiGANPeriodDiscriminator(tf.keras.layers.Layer):
    """Tensorflow Hifigan period discriminator module."""

    def __init__(
        self,
        period,
        out_channels=1,
        n_layers=5,
        kernel_size=5,
        strides=3,
        filters=8,
        filter_scales=4,
        max_filters=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        initializer_seed=42,
        is_weight_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.period = period
        self.out_filters = out_channels
        self.convs = []

        for i in range(n_layers):
            self.convs.append(
                tf.keras.layers.Conv2D(
                    filters=min(filters * (filter_scales ** (i + 1)), max_filters),
                    kernel_size=(kernel_size, 1),
                    strides=(strides, 1),
                    padding="same",
                )
            )
        self.conv_post = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(3, 1), padding="same",
        )
        self.activation = getattr(tf.keras.layers, nonlinear_activation)(
            **nonlinear_activation_params
        )

        if is_weight_norm:
            self._apply_weightnorm(self.convs)
            self.conv_post = WeightNormalization(self.conv_post)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of output tensors.
        """
        shape = tf.shape(x)
        n_pad = tf.convert_to_tensor(0, dtype=tf.int32)
        if shape[1] % self.period != 0:
            n_pad = self.period - (shape[1] % self.period)
            x = tf.pad(x, [[0, 0], [0, n_pad], [0, 0]], "REFLECT")
        x = tf.reshape(
            x, [shape[0], (shape[1] + n_pad) // self.period, self.period, x.shape[2]]
        )
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
        x = self.conv_post(x)
        x = tf.reshape(x, [shape[0], -1, self.out_filters])
        return [x]

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass


MODEL_FILE_NAME = 'MY_MOODEL_TEST'

class BaseModel(tf.keras.Model):
    def set_config(self, config):
        self.config = config

    def save_pretrained(self, saved_path):
        """Save config and weights to file"""
        os.makedirs(saved_path, exist_ok=True)
        self.config.save_pretrained(saved_path)
        self.save_weights(os.path.join(saved_path, MODEL_FILE_NAME))



class TFHifiGANMultiPeriodDiscriminator(BaseModel):
    """Tensorflow Hifigan Multi Period discriminator module."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(len(config.period_scales)):
            self.discriminator += [
                TFHifiGANPeriodDiscriminator(
                    config.period_scales[i],
                    out_channels=config.out_channels,
                    n_layers=config.n_layers,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    filters=config.filters,
                    filter_scales=config.filter_scales,
                    max_filters=config.max_filters,
                    nonlinear_activation=config.nonlinear_activation,
                    nonlinear_activation_params=config.nonlinear_activation_params,
                    initializer_seed=config.initializer_seed,
                    is_weight_norm=config.is_weight_norm,
                    name="hifigan_period_discriminator_._{}".format(i),
                )
            ]

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: list of each discriminator outputs
        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
        return outs




class TFMelGANDiscriminator(tf.keras.layers.Layer):
    """Tensorflow MelGAN generator module."""

    def __init__(
        self,
        out_channels=1,
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
        is_weight_norm=True,
        initializer_seed=0.02,
        **kwargs
    ):
        """Initilize MelGAN discriminator module.
        Args:
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15.
                the last two layers' kernel size will be 5 and 3, respectively.
            filters (int): Initial number of filters for conv layer.
            max_downsample_filters (int): Maximum number of filters for downsampling layers.
            use_bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            padding_type (str): Padding type (support only "REFLECT", "CONSTANT", "SYMMETRIC")
        """
        super().__init__(**kwargs)
        discriminator = []

        # check kernel_size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        discriminator = [
            TFReflectionPad1d(
                (np.prod(kernel_sizes) - 1) // 2, padding_type=padding_type
            ),
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=int(np.prod(kernel_sizes)),
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            ),
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
        ]

        # add downsample layers
        in_chs = filters
        with tf.keras.utils.CustomObjectScope({"GroupConv1D": GroupConv1D}):
            for downsample_scale in downsample_scales:
                out_chs = min(in_chs * downsample_scale, max_downsample_filters)
                discriminator += [
                    GroupConv1D(
                        filters=out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        strides=downsample_scale,
                        padding="same",
                        use_bias=use_bias,
                        groups=in_chs // 4,
                        kernel_initializer=get_initializer(initializer_seed),
                    )
                ]
                discriminator += [
                    getattr(tf.keras.layers, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                ]
                in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_filters)
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_chs,
                kernel_size=kernel_sizes[0],
                padding="same",
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            )
        ]
        discriminator += [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            )
        ]
        discriminator += [
            tf.keras.layers.Conv1D(
                filters=out_channels,
                kernel_size=kernel_sizes[1],
                padding="same",
                use_bias=use_bias,
                kernel_initializer=get_initializer(initializer_seed),
            )
        ]

        if is_weight_norm is True:
            self._apply_weightnorm(discriminator)

        self.disciminator = discriminator

    def call(self, x, **kwargs):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.disciminator:
            x = f(x)
            outs += [x]
        return outs

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass





class TFMelGANMultiScaleDiscriminator(BaseModel):
    """MelGAN multi-scale discriminator module."""

    def __init__(self, config, **kwargs):
        """Initilize MelGAN multi-scale discriminator module.
        Args:
            config: config object for melgan discriminator
        """
        super().__init__(**kwargs)
        self.discriminator = []

        # add discriminator
        for i in range(config.scales):
            self.discriminator += [
                TFMelGANDiscriminator(
                    out_channels=config.out_channels,
                    kernel_sizes=config.kernel_sizes,
                    filters=config.filters,
                    max_downsample_filters=config.max_downsample_filters,
                    use_bias=config.use_bias,
                    downsample_scales=config.downsample_scales,
                    nonlinear_activation=config.nonlinear_activation,
                    nonlinear_activation_params=config.nonlinear_activation_params,
                    padding_type=config.padding_type,
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name="melgan_discriminator_scale_._{}".format(i),
                )
            ]
            self.pooling = getattr(tf.keras.layers, config.downsample_pooling)(
                **config.downsample_pooling_params
            )

    def call(self, x, **kwargs):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, T, 1).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminator:
            outs += [f(x)]
            x = self.pooling(x)
        return outs




CONFIG_FILE_NAME = "config.yml"
class BaseConfig(abc.ABC):
    def set_config_params(self, config_params):
        self.config_params = config_params

    def save_pretrained(self, saved_path):
        """Save config to file"""
        os.makedirs(saved_path, exist_ok=True)
        with open(os.path.join(saved_path, CONFIG_FILE_NAME), "w") as file:
            yaml.dump(self.config_params, file, Dumper=yaml.Dumper)




class HifiGANGeneratorConfig(BaseConfig):
    """Initialize HifiGAN Generator Config."""

    def __init__(
        self,
        out_channels=1,
        kernel_size=7,
        filters=128,
        use_bias=True,
        upsample_scales=[8, 8, 2, 2],
        stacks=3,
        stack_kernel_size=[3, 7, 11],
        stack_dilation_rate=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
        use_final_nolinear_activation=True,
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for HifiGAN Generator model."""
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.use_bias = use_bias
        self.upsample_scales = upsample_scales
        self.stacks = stacks
        self.stack_kernel_size = stack_kernel_size
        self.stack_dilation_rate = stack_dilation_rate
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.use_final_nolinear_activation = use_final_nolinear_activation
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed



class HifiGANDiscriminatorConfig(object):
    """Initialize HifiGAN Discriminator Config."""

    def __init__(
        self,
        out_channels=1,
        period_scales=[2, 3, 5, 7, 11],
        n_layers=5,
        kernel_size=5,
        strides=3,
        filters=8,
        filter_scales=4,
        max_filters=1024,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for MelGAN Discriminator model."""
        self.out_channels = out_channels
        self.period_scales = period_scales
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.filter_scales = filter_scales
        self.max_filters = max_filters
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed

class MelGANDiscriminatorConfig(object):
    """Initialize MelGAN Discriminator Config."""

    def __init__(
        self,
        out_channels=1,
        scales=3,
        downsample_pooling="AveragePooling1D",
        downsample_pooling_params={"pool_size": 4, "strides": 2,},
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT",
        is_weight_norm=True,
        initializer_seed=42,
        **kwargs
    ):
        """Init parameters for MelGAN Discriminator model."""
        self.out_channels = out_channels
        self.scales = scales
        self.downsample_pooling = downsample_pooling
        self.downsample_pooling_params = downsample_pooling_params
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.max_downsample_filters = max_downsample_filters
        self.use_bias = use_bias
        self.downsample_scales = downsample_scales
        self.nonlinear_activation = nonlinear_activation
        self.nonlinear_activation_params = nonlinear_activation_params
        self.padding_type = padding_type
        self.is_weight_norm = is_weight_norm
        self.initializer_seed = initializer_seed


class TFHifiGANDiscriminator(tf.keras.Model):
    def __init__(self, multiperiod_dis, multiscale_dis, **kwargs):
        super().__init__(**kwargs)
        self.multiperiod_dis = multiperiod_dis
        self.multiscale_dis = multiscale_dis

    def call(self, x):
        outs = []
        period_outs = self.multiperiod_dis(x)
        scale_outs = self.multiscale_dis(x)
        outs.extend(period_outs)
        outs.extend(scale_outs)
        return outs



generator = TFHifiGANGenerator(
    HifiGANGeneratorConfig(**config["hifigan_generator_params"]),
    name="hifigan_generator",
)

multiperiod_discriminator = TFHifiGANMultiPeriodDiscriminator(
    HifiGANDiscriminatorConfig(**config["hifigan_discriminator_params"]),
    name="hifigan_multiperiod_discriminator",
)
multiscale_discriminator = TFMelGANMultiScaleDiscriminator(
    MelGANDiscriminatorConfig(
        **config["melgan_discriminator_params"],
        name="melgan_multiscale_discriminator",
    )
)

discriminator = TFHifiGANDiscriminator(
    multiperiod_discriminator,
    multiscale_discriminator,
    name="hifigan_discriminator",
)

       

class TFMelSpectrogram(tf.keras.layers.Layer):
    """Mel Spectrogram loss."""

    def __init__(
        self,
        n_mels=80,
        f_min=80.0,
        f_max=7600,
        frame_length=1024,
        frame_step=256,
        fft_length=1024,
        sample_rate=16000,
        **kwargs
    ):
        """Initialize."""
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            n_mels, fft_length // 2 + 1, sample_rate, f_min, f_max
        )

    def _calculate_log_mels_spectrogram(self, signals):
        """Calculate forward propagation.
        Args:
            signals (Tensor): signal (B, T).
        Returns:
            Tensor: Mel spectrogram (B, T', 80)
        """
        stfts = tf.signal.stft(
            signals,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
        )
        linear_spectrograms = tf.abs(stfts)
        mel_spectrograms = tf.tensordot(
            linear_spectrograms, self.linear_to_mel_weight_matrix, 1
        )
        mel_spectrograms.set_shape(
            linear_spectrograms.shape[:-1].concatenate(
                self.linear_to_mel_weight_matrix.shape[-1:]
            )
        )
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)  # prevent nan.
        return log_mel_spectrograms

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Mean absolute Error Spectrogram Loss.
        """
        y_mels = self._calculate_log_mels_spectrogram(y)
        x_mels = self._calculate_log_mels_spectrogram(x)
        return tf.reduce_mean(
            tf.abs(y_mels - x_mels), axis=list(range(1, len(x_mels.shape)))
        )




class MelganTrainer(GanBasedTrainer):
    """Melgan Trainer class based on GanBasedTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.


        """
        super(MelganTrainer, self).__init__(
            steps,
            epochs,
            #config,
            strategy,
            is_generator_mixed_precision,
            is_discriminator_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "mels_spectrogram_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mels_loss = TFMelSpectrogram()

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        audios = batch["audios"]
        y_hat = outputs

        p_hat = self._discriminator(y_hat)
        p = self._discriminator(tf.expand_dims(audios, 2))
        adv_loss = 0.0
        for i in range(len(p_hat)):
            adv_loss += calculate_3d_loss(
                tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
            )
        adv_loss /= i + 1

        # define feature-matching loss
        fm_loss = 0.0
        for i in range(len(p_hat)):
            for j in range(len(p_hat[i]) - 1):
                fm_loss += calculate_3d_loss(
                    p[i][j], p_hat[i][j], loss_fn=self.mae_loss
                )
        fm_loss /= (i + 1) * (j + 1)
        adv_loss += self.config["lambda_feat_match"] * fm_loss

        per_example_losses = adv_loss

        dict_metrics_losses = {
            "adversarial_loss": adv_loss,
            "fm_loss": fm_loss,
            "gen_loss": adv_loss,
            "mels_spectrogram_loss": calculate_2d_loss(
                audios, tf.squeeze(y_hat, -1), loss_fn=self.mels_loss
            ),
        }

        return per_example_losses, dict_metrics_losses

    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        audios = batch["audios"]
        y_hat = gen_outputs

        y = tf.expand_dims(audios, 2)
        p = self._discriminator(y)
        p_hat = self._discriminator(y_hat)

        real_loss = 0.0
        fake_loss = 0.0
        for i in range(len(p)):
            real_loss += calculate_3d_loss(
                tf.ones_like(p[i][-1]), p[i][-1], loss_fn=self.mse_loss
            )
            fake_loss += calculate_3d_loss(
                tf.zeros_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
            )
        real_loss /= i + 1
        fake_loss /= i + 1
        dis_loss = real_loss + fake_loss

        # calculate per_example_losses and dict_metrics_losses
        per_example_losses = dis_loss

        dict_metrics_losses = {
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "dis_loss": dis_loss,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # generate
        y_batch_ = self.one_step_predict(batch)
        y_batch = batch["audios"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            y_batch_ = y_batch_.values[0].numpy()
            y_batch = y_batch.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            y_batch_ = y_batch_.numpy()
            y_batch = y_batch.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 0):
            # convert to ndarray
            y, y_ = tf.reshape(y, [-1]).numpy(), tf.reshape(y_, [-1]).numpy()

            # plit figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavefile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )




# dummy input to build model.
fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
y_hat = generator(fake_mels)
discriminator(y_hat)


# generator.summary()
# discriminator.summary()

# define optimizer
generator_lr_fn = getattr(
    tf.keras.optimizers.schedules, config["generator_optimizer_params"]["lr_fn"]
)(**config["generator_optimizer_params"]["lr_params"])
discriminator_lr_fn = getattr(
    tf.keras.optimizers.schedules,
    config["discriminator_optimizer_params"]["lr_fn"],
)(**config["discriminator_optimizer_params"]["lr_params"])

gen_optimizer = tf.keras.optimizers.Adam(
    learning_rate=generator_lr_fn,
    amsgrad=config["generator_optimizer_params"]["amsgrad"],
)
dis_optimizer = tf.keras.optimizers.Adam(
    learning_rate=discriminator_lr_fn,
    amsgrad=config["discriminator_optimizer_params"]["amsgrad"],
)



def return_strategy():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(physical_devices) == 1:
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        return tf.distribute.MirroredStrategy()


def calculate_3d_loss(y_gt, y_pred, loss_fn):
    """Calculate 3d loss, normally it's mel-spectrogram loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    if y_gt_T > y_pred_T:
        y_gt = tf.slice(y_gt, [0, 0, 0], [-1, y_pred_T, -1])
    elif y_pred_T > y_gt_T:
        y_pred = tf.slice(y_pred, [0, 0, 0], [-1, y_gt_T, -1])

    loss = loss_fn(y_gt, y_pred)
    if isinstance(loss, tuple) is False:
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))  # shape = [B]
    else:
        loss = list(loss)
        for i in range(len(loss)):
            loss[i] = tf.reduce_mean(
                loss[i], list(range(1, len(loss[i].shape)))
            )  # shape = [B]
    return loss


def calculate_2d_loss(y_gt, y_pred, loss_fn):
    """Calculate 2d loss, normally it's durrations/f0s/energys loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    if y_gt_T > y_pred_T:
        y_gt = tf.slice(y_gt, [0, 0], [-1, y_pred_T])
    elif y_pred_T > y_gt_T:
        y_pred = tf.slice(y_pred, [0, 0], [-1, y_gt_T])

    loss = loss_fn(y_gt, y_pred)
    if isinstance(loss, tuple) is False:
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))  # shape = [B]
    else:
        loss = list(loss)
        for i in range(len(loss)):
            loss[i] = tf.reduce_mean(
                loss[i], list(range(1, len(loss[i].shape)))
            )  # shape = [B]

    return loss

class TFSpectralConvergence(tf.keras.layers.Layer):
    """Spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / tf.norm(
            y_mag, ord="fro", axis=(-2, -1)
        )


class TFLogSTFTMagnitude(tf.keras.layers.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.abs(tf.math.log(y_mag) - tf.math.log(x_mag))





class TFSTFT(tf.keras.layers.Layer):
    """STFT loss module."""

    def __init__(self, frame_length=600, frame_step=120, fft_length=1024):
        """Initialize."""
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.spectral_convergenge_loss = TFSpectralConvergence()
        self.log_stft_magnitude_loss = TFLogSTFTMagnitude()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value (pre-reduce).
            Tensor: Log STFT magnitude loss value (pre-reduce).
        """
        x_mag = tf.abs(
            tf.signal.stft(
                signals=x,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )
        y_mag = tf.abs(
            tf.signal.stft(
                signals=y,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )

        # add small number to prevent nan value.
        # compatible with pytorch version.
        x_mag = tf.clip_by_value(tf.math.sqrt(x_mag ** 2 + 1e-7), 1e-7, 1e3)
        y_mag = tf.clip_by_value(tf.math.sqrt(y_mag ** 2 + 1e-7), 1e-7, 1e3)

        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)
        mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

        return sc_loss, mag_loss


class TFMultiResolutionSTFT(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_lengths=[1024, 2048, 512],
        frame_lengths=[600, 1200, 240],
        frame_steps=[120, 240, 50],
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            frame_lengths (list): List of FFT sizes.
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of window lengths.
        """
        super().__init__()
        assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
        self.stft_losses = []
        for frame_length, frame_step, fft_length in zip(
            frame_lengths, frame_steps, fft_lengths
        ):
            self.stft_losses.append(TFSTFT(frame_length, frame_step, fft_length))

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(y, x)
            sc_loss += tf.reduce_mean(sc_l, axis=list(range(1, len(sc_l.shape))))
            mag_loss += tf.reduce_mean(mag_l, axis=list(range(1, len(mag_l.shape))))

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class MultiSTFTMelganTrainer(MelganTrainer):
    """Multi STFT Melgan Trainer class based on MelganTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.

        """
        super(MultiSTFTMelganTrainer, self).__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            strategy=strategy,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "spectral_convergence_loss",
            "log_magnitude_loss",
        ]

        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.stft_loss = TFMultiResolutionSTFT(**self.config["stft_loss_params"])

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        dict_metrics_losses = {}
        per_example_losses = 0.0

        audios = batch["audios"]
        y_hat = outputs

        # calculate multi-resolution stft loss
        sc_loss, mag_loss = calculate_2d_loss(
            audios, tf.squeeze(y_hat, -1), self.stft_loss
        )

        # trick to prevent loss expoded here
        sc_loss = tf.where(sc_loss >= 15.0, 0.0, sc_loss)
        mag_loss = tf.where(mag_loss >= 15.0, 0.0, mag_loss)

        # compute generator loss
        gen_loss = 0.5 * (sc_loss + mag_loss)

        if self.steps >= self.config["discriminator_train_start_steps"]:
            p_hat = self._discriminator(y_hat)
            p = self._discriminator(tf.expand_dims(audios, 2))
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += calculate_3d_loss(
                    tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
                )
            adv_loss /= i + 1

            # define feature-matching loss
            fm_loss = 0.0
            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    fm_loss += calculate_3d_loss(
                        p[i][j], p_hat[i][j], loss_fn=self.mae_loss
                    )
            fm_loss /= (i + 1) * (j + 1)
            adv_loss += self.config["lambda_feat_match"] * fm_loss
            gen_loss += self.config["lambda_adv"] * adv_loss

            dict_metrics_losses.update({"adversarial_loss": adv_loss})
            dict_metrics_losses.update({"fm_loss": fm_loss})

        dict_metrics_losses.update({"gen_loss": gen_loss})
        dict_metrics_losses.update({"spectral_convergence_loss": sc_loss})
        dict_metrics_losses.update({"log_magnitude_loss": mag_loss})

        per_example_losses = gen_loss
        return per_example_losses, dict_metrics_losses




# define trainer
trainer = MultiSTFTMelganTrainer(
    steps=0,
    epochs=0,
    config=config,
    strategy=STRATEGY,
    #is_generator_mixed_precision=args.generator_mixed_precision,
    #is_discriminator_mixed_precision=args.discriminator_mixed_precision,
)



IM_SIZE = (40,87) 
BATCH_SIZE = 128



if __name__ == "__main__":  
    pross = PreprocessData(frame_length=100, frame_mel=80)
    X = pross.process()
    X = np.squeeze(X)
    X = np.moveaxis(X, 1,2)
    X_train, X_test = train_test_split(X, test_size=0.05, random_state = 42)
    
    train_dataset = X_train
    valid_dataset = X_test

    trainer.compile(
    gen_model=generator,
    dis_model=discriminator,
    gen_optimizer=gen_optimizer,
    dis_optimizer=dis_optimizer,
    )
    
    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            #resume=args.resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
    
    
    
    
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        