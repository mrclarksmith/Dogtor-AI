import os
import pickle

from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda 
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, binary_crossentropy, mse
import numpy as np
import tensorflow as tf
import datetime 

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# tf.compat.v1.disable_eager_execution()
#https://github.com/musikalkemist/generating-sound-with-neural-networks/blob/main/14%20Sound%20generation%20with%20VAE/code/autoencoder.py

class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000.
        self.loss = mse
        self.gamma = 1.0
        self.capacity = 0.0
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        self.model_output = None
        self.bottleneck = None
        
        self.mu = None
        self.log_variance = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        print("model summary")
        self.model.summary()
        print("end of summary")

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        # self.model.add_loss(self.vae_loss(self._model_input, self.model_output))
        self.model.compile(optimizer=optimizer,
                           loss=self.VAE_loss_fn,
                           # metrics=['MeanSquaredError', 'KLDivergence']
                           )
    
    def vae_loss(self, inputs, outputs):
        z, z_mean, z_log_var = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        
        # mse_loss = self.mse_loss_fn(inputs, reconstructed)
        
        
        error = tf.subtract(inputs,reconstructed)
        mse_loss = K.mean(K.square(error), axis=[1, 2, 3])
        
        
        total_loss =  kl_loss + mse_loss  * self.reconstruction_loss_weight
        # self.add_loss(total_loss)
        return total_loss
        #return self._calculate_combined_loss(inputs, outputs)
    
    def train(self, x_train, batch_size, num_epochs):
        #model_checkpoint_callback = self._callback(batch_size)
                         
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=False,
                       #callbacks = [model_checkpoint_callback]
                       )

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        
    def save_model(self, save_folder=".", f="tf"):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_model(save_folder, f=f)
        
        
    def _save_model(self, save_folder=".",f="tf"):
        save_path = os.path.join(save_folder, f"full_model.{f}")
        self.model.save(save_path,save_format=f)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)[1]
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def load_model_h5(self,save_folder):
        print(save_folder)
        save_path = os.path.join(save_folder, "full_model.h5")
        print(save_path)
        self.model = tf.keras.models.load_model(save_path, 
                                   custom_objects={'L__sample_point_from_normal_distribution': self.L__sample_point_from_normal_distribution})
        
    def load_model_tf(self,save_folder):
        print(save_folder)
        save_path = os.path.join(save_folder, "full_model.tf")
        print(save_path)
        self.model = tf.keras.models.load_model(save_path, 
                                   custom_objects={'L__sample_point_from_normal_distribution': self.L__sample_point_from_normal_distribution})



    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        print(parameters)
        autoencoder = VAE(*parameters)

        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder
    
    # def _callback(self, batch_size):
    #     callback_save_path = os.path.join("checkpoint_save","checkpoint_weights.h5")
    #     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #             filepath=callback_save_path,
    #             save_weights_only=True,
    #             monitor=self._calculate_combined_loss,
    #             mode='auto',
    #             save_freq=batch_size,
    #             save_best_only=True,
    #             )
    #     return model_checkpoint_callback
    
#tf function losses
    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = tf.math.add(tf.math.multiply(self.reconstruction_loss_weight ,reconstruction_loss), kl_loss)
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = tf.subtract(y_target,y_predicted)
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = tf.math.multiply(tf.math.negative(0.5), tf.math.reduce_sum(tf.math.add(1.,
                                                  tf.math.add_n([self.log_variance, 
                                                    tf.math.negative(K.square(self.mu)), 
                                                    tf.math.negative(K.exp(self.log_variance)),
                                                          ])
                                                ), axis=1
                                            )
                              ,name="_calculate_k1_loss_mult")
        print(kl_loss)
        return kl_loss


    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)



    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()


    def _build_autoencoder(self):
        model_input = self._model_input
        self._encoder_output = self.encoder(model_input)
        # self.mu_out = self.encoder(model_input)[1]
        # self.log_out = self.encoder(model_input)[2]
        self.model_output = self.decoder(self._encoder_output)
        # output = [self.model_output, self.mu_out, self.log_out]
        self.model = Model(self._model_input, self.model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        print("d1")
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer
#build encoder
    def _build_encoder(self):
        self._model_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(self._model_input)
        z, self.mu, self.log_variance = self._add_bottleneck(conv_layers)

        self.encoder = Model(self._model_input, z, name="encoder")
        print("e1")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index #+ 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

#f
    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        print(self._shape_before_bottleneck, "shape before bottleneck")
        x = Flatten()(x)
        mu = Dense(self.latent_space_dim, name="mu")(x)
        
        log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        #https://errorsfixing.com/custom-keras-layer-fails-with-typeerror-could-not-build-a-typespec-for-kerastensor/
        #https://www.programcreek.com/python/?code=yzhao062%2Fpyod%2Fpyod-master%2Fpyod%2Fmodels%2Fvae.py

        print("t1")
        x = self.L__sample_point_from_normal_distribution(name="sample_point_lambda")([mu, log_variance])
        print("t3")
        return x, mu, log_variance
    
    class L__sample_point_from_normal_distribution(tf.keras.layers.Layer):
        def __init__(self,**kwargs):
            # super().__init__( **kwargs)
            super().__init__(name="sample_point_lambda")
            
        def call(self, args, training=None, mask=None):
            mu, log_variance = args
            batch = K.shape(mu)[0]
            dim = K.int_shape(mu)[1]
            epsilon = K.random_normal(shape=(batch, dim), mean=0.,
                                      stddev=1.)
            sampled_point = tf.math.add(mu , tf.math.multiply( 
                tf.math.exp( tf.math.divide(log_variance, 2.)) , epsilon))
            return sampled_point
        
        def get_config(self):
            config = super().get_config()
            # config.update({"units": self.units})
            return config

    def VAE_loss_fn(self, inputs, outputs):
        print(outputs, "ooutputs shape", inputs.shape, "inputs shape")
        reconstructed, z_mean, z_log_var = outputs, self.mu, self.log_variance
        
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        
        # mse_loss = self.mse_loss_fn(inputs, reconstructed)
        
        
        error = tf.subtract(inputs,reconstructed)
        mse_loss = K.mean(K.square(error), axis=[1, 2, 3])
        
        
        total_loss =  kl_loss + mse_loss  * self.reconstruction_loss_weight
        # self.add_loss(total_loss)
        return total_loss

    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
      
      
        
    def compute_loss(self,model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
      
    # class VAE_loss_fn(tf.keras.losses.Loss):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         # super().__init__(name="VAE_loss_fn")
            
    #     def call(self, inputs, outputs):
    #         print(outputs, "ooutputs shape", inputs.shape, "inputs shape")
    #         reconstructed, z_mean, z_log_var = outputs
            
    #         # Add KL divergence regularization loss.
    #         kl_loss = -0.5 * tf.reduce_mean(
    #             z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            
    #         # mse_loss = self.mse_loss_fn(inputs, reconstructed)
            
            
    #         error = tf.subtract(inputs,reconstructed)
    #         mse_loss = K.mean(K.square(error), axis=[1, 2, 3])
            
            
    #         total_loss =  kl_loss + mse_loss  * self.reconstruction_loss_weight
    #         # self.add_loss(total_loss)
    #         return total_loss
        
        # def get_config(self):
        #     config = super().get_config()
        #     # config.update({"units": self.units})
        #     return config




 
class VAE_model(Model):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super(VAE_model, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.reconstruction_loss_weight = 1000000
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()


    def call(self, inputs, states=None, return_state=False, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        
        mse_loss = self.mse_loss_fn(inputs, reconstructed)
        total_loss =  kl_loss + mse_loss  * self.reconstruction_loss_weight
        self.add_loss(total_loss)


        # self.add_metric(total_loss, name="total_loss")
        self.add_metric(mse_loss, name="mse_loss")
        self.add_metric(kl_loss, name="kl_loss")
        return reconstructed





if __name__ == "__main__":
    #demo encoder
    # autoencoder = VAE(
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=2
    # )
    # autoencoder.summary()

    pass

