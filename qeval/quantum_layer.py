import cirq
import numpy as np
import tensorflow as tf  # type: ignore
import tensorflow_quantum as tfq  # type: ignore
from tensorflow.keras import initializers  # type: ignore


class Input_Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_symbols,
        n_input,
        activation=tf.math.atan,
        trainable_input=True,
        input_type="data",
        specific_training=False,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.input_symbols = input_symbols
        self.n_input = n_input
        self.activation = activation
        self.input_type = input_type
        self.specific_training = specific_training

        if specific_training:
            self.modify = tf.convert_to_tensor([[1, 1, 0]], dtype=np.float32)
            self.keep = tf.convert_to_tensor([[0, 0, 1]], dtype=np.float32)

            if self.input_type == "data":
                self.tiled_keep = tf.tile(
                    self.keep,
                    multiples=[1, int(len(self.input_symbols) / self.n_input)],
                )
                self.tiled_modify = tf.tile(
                    self.modify,
                    multiples=[1, int(len(self.input_symbols) / self.n_input)],
                )
            else:
                self.tiled_keep = tf.repeat(
                    self.keep,
                    repeats=int(len(self.input_symbols) / self.n_input),
                    axis=1,
                )
                self.tiled_modify = tf.repeat(
                    self.modify,
                    repeats=int(len(self.input_symbols) / self.n_input),
                    axis=1,
                )

        self.input_parameters = self.add_weight(
            "input_parameters",
            shape=(len(self.input_symbols),),
            initializer=tf.constant_initializer(1),
            dtype=tf.float32,
            trainable=trainable_input,
        )

    def call(self, inputs):
        tensor_input = tf.convert_to_tensor(inputs, dtype=np.float32)

        if self.input_type == "data":
            tiled_input = tf.tile(
                tensor_input, multiples=[1, int(len(self.input_symbols) / self.n_input)]
            )
        else:
            tiled_input = tf.repeat(
                tensor_input,
                repeats=int(len(self.input_symbols) / self.n_input),
                axis=1,
            )

        if self.specific_training:
            modify_inputs = tf.math.multiply(tiled_input, self.tiled_modify)
            keep_inputs = tf.math.multiply(tiled_input, self.tiled_keep)
            scaled_modify_inputs = tf.einsum(
                "i,ji->ji", self.input_parameters, modify_inputs
            )
            activated_modify_inputs = tf.keras.layers.Activation(self.activation)(
                scaled_modify_inputs
            )
            all_inputs = tf.math.add(keep_inputs, activated_modify_inputs)

        else:
            scaled_inputs = tf.einsum("i,ji->ji", self.input_parameters, tiled_input)
            all_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        return all_inputs


class PQC_customized(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        input_symbols,
        circuit_symbols,
        operators,
        initializer=initializers.RandomUniform(0, np.pi),
        **kwargs
    ):

        super().__init__(**kwargs)
        self.input_symbols = input_symbols
        self.circuit_symbols = circuit_symbols
        self.initializer = tf.keras.initializers.get(initializer)
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(model_circuit, operators)

        symbols = [str(symb) for symb in input_symbols + circuit_symbols]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])

        self.circuit_parameters = self.add_weight(
            "circuit_parameters",
            shape=(len(self.circuit_symbols),),
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=np.pi),
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs), 0)

        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_parameters = tf.tile(
            [self.circuit_parameters], multiples=[batch_dim, 1]
        )

        tiled_all_parameters = tf.concat([inputs, tiled_up_parameters], axis=1)
        joined_parameters = tf.gather(tiled_all_parameters, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_parameters])


class Output_Layer(tf.keras.layers.Layer):
    def __init__(self, units, rescale=False, trainable_output=True, **kwargs):

        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.rescale = rescale

        self.factors = self.add_weight(
            "output_parameters",
            shape=(1, self.units),
            initializer="ones",
            trainable=trainable_output,
        )

    def call(self, inputs):
        if self.rescale:
            return tf.math.multiply(
                (inputs + 1) / 2,
                tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0),
            )
        else:
            return tf.math.multiply(
                inputs, tf.repeat(self.factors, repeats=tf.shape(inputs)[0], axis=0)
            )
