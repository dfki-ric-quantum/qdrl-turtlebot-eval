import tensorflow as tf  # type: ignore
from tf_agents.networks import sequential  # type: ignore

from qeval.quantum_circuit import create_reupload_circuit
from qeval.quantum_layer import Input_Layer, Output_Layer, PQC_customized
from qeval.quantum_measurement import create_measurement


def create_classical_model(n_actions, layers):
    dense_layers = [
        tf.keras.layers.Dense(num_units, activation="relu") for num_units in layers
    ]

    q_values_layer = tf.keras.layers.Dense(n_actions, activation="linear")

    return sequential.Sequential(dense_layers + [q_values_layer])


def create_quantum_model(
    n_qubits=3,
    layers=5,
    rot_per_unitary=3,
    input_style=3,
    data_reupload=True,
    trainable_input=True,
    zero_layer=True,
    rescale=False,
    activition=tf.math.atan,
    n_states=3,
    n_actions=3,
    env="qturtle",
    trainable_output=True,
):

    input_symbols, trainable_symbols, circuit = create_reupload_circuit(
        num_qubits=n_qubits,
        layers=layers,
        rot_per_unitary=rot_per_unitary,
        input_style=input_style,
        data_reupload=data_reupload,
        trainable_input=trainable_input,
        zero_layer=zero_layer
    )

    measurement = create_measurement(num_qubits=n_qubits, env=env)

    input_layer = Input_Layer(
        input_symbols=input_symbols,
        n_input=n_states,
        trainable_input=trainable_input,
        activation=activition,
        input_type="data",
        name="specific_input",
    )

    circuit_layer = PQC_customized(
        model_circuit=circuit,
        input_symbols=input_symbols,
        circuit_symbols=trainable_symbols,
        operators=measurement,
        name="pqc_circuit",
    )

    output_layer = Output_Layer(
        units=n_actions,
        rescale=rescale,
        trainable_output=trainable_output,
        name="specific_output",
    )

    return sequential.Sequential([input_layer] + [circuit_layer] + [output_layer])
