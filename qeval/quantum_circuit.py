import cirq
import sympy  # type: ignore


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def get_new_param(symbol_name, qubit, position, layer=None):
    """
    return new learnable parameter
    """
    if str(layer):
        new_param = sympy.symbols(
            symbol_name + "_" + str(qubit) + "_" + str(layer) + "_" + str(position)
        )
    else:
        new_param = sympy.symbols(symbol_name + "_" + str(qubit) + "_" + str(position))

    return new_param


def create_input(qubit, n_qubit, symbol_name, layer=None, input_style=1):
    circuit = cirq.Circuit()

    input_parameters = []

    if input_style == 1:
        input_parameter = get_new_param(symbol_name, n_qubit, 0, layer)
        circuit += cirq.rx(input_parameter).on(qubit)
        input_parameters.append(input_parameter)

    elif input_style == 2:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)

        circuit += cirq.rx(input_parameter_0).on(qubit)
        circuit += cirq.ry(input_parameter_1).on(qubit)

        input_parameters.append(input_parameter_0)
        input_parameters.append(input_parameter_1)

    elif input_style == 3:
        input_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        input_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        input_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit += cirq.rx(input_parameter_0).on(qubit)
        circuit += cirq.ry(input_parameter_1).on(qubit)
        circuit += cirq.rx(input_parameter_2).on(qubit)

        input_parameters.append(input_parameter_0)
        input_parameters.append(input_parameter_1)
        input_parameters.append(input_parameter_2)
    else:
        raise ValueError("input_style not implemented.{}".format(input_style))

    return circuit, to_tuple(input_parameters)


def create_entanglement(num_qubits):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)

    for i in range(num_qubits - 1):
        circuit += cirq.CZ(qubits[i], qubits[i + 1])

    if num_qubits > 2:
        circuit += cirq.CZ(qubits[-1], qubits[0])

    return circuit


def create_unitary(qubit, n_qubit, layer, rot_per_unitary=3, circuit_type="skolik"):
    circuit = cirq.Circuit()
    body_parameters = []
    symbol_name = "train"

    if rot_per_unitary == 2:
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)

        circuit += cirq.ry(body_parameter_0).on(qubit)
        circuit += cirq.rz(body_parameter_1).on(qubit)

        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)

    elif rot_per_unitary == 3 and circuit_type == "skolik":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit += cirq.rx(body_parameter_0).on(qubit)
        circuit += cirq.ry(body_parameter_1).on(qubit)
        circuit += cirq.rz(body_parameter_2).on(qubit)

        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)
        body_parameters.append(body_parameter_2)

    elif rot_per_unitary == 3 and circuit_type == "han":
        body_parameter_0 = get_new_param(symbol_name, n_qubit, 0, layer)
        body_parameter_1 = get_new_param(symbol_name, n_qubit, 1, layer)
        body_parameter_2 = get_new_param(symbol_name, n_qubit, 2, layer)

        circuit += cirq.rx(body_parameter_0).on(qubit)
        circuit += cirq.ry(body_parameter_1).on(qubit)
        circuit += cirq.rx(body_parameter_2).on(qubit)

        body_parameters.append(body_parameter_0)
        body_parameters.append(body_parameter_1)
        body_parameters.append(body_parameter_2)

    else:
        raise ValueError("rot_per_unitary =/= (2,3) not implemented")

    return circuit, to_tuple(body_parameters)


def create_reupload_circuit(
    num_qubits,
    layers,
    rot_per_unitary=3,
    input_style=3,
    data_reupload=True,
    trainable_input=True,
    zero_layer=True,
):

    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)

    input_symbols_name = "in"
    if trainable_input:
        input_symbols_name = "in_train"

    input_symbols = ()
    trainable_symbols = ()

    for layer in range(layers + 1):

        if layer == 0 and zero_layer:
            for i, qubit in enumerate(qubits):
                circuit_unitary, unitary_parameters = create_unitary(
                    qubit, i, layer, rot_per_unitary
                )
                circuit += circuit_unitary
                trainable_symbols += unitary_parameters

            circuit += create_entanglement(num_qubits)

        elif layer > 0:
            if layer == 1 or data_reupload:
                for i, qubit in enumerate(qubits):
                    input_circuit, input_parameters = create_input(
                        qubit, i, input_symbols_name, layer, input_style
                    )
                    circuit += input_circuit
                    input_symbols += input_parameters

            for i, qubit in enumerate(qubits):
                circuit_unitary, unitary_parameters = create_unitary(
                    qubit, i, layer, rot_per_unitary
                )
                circuit += circuit_unitary
                trainable_symbols += unitary_parameters

            circuit += create_entanglement(num_qubits)

    return input_symbols, trainable_symbols, circuit
