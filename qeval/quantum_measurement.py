import cirq


def create_measurement(num_qubits=3, env='qturtle'):
    qubits = cirq.GridQubit.rect(1, num_qubits)
    measurement = []

    if env == 'FrozenLake-v1' and num_qubits == 4:
        measurement.append(cirq.Z(qubits[0]))
        measurement.append(cirq.Z(qubits[1]))
        measurement.append(cirq.Z(qubits[2]))
        measurement.append(cirq.Z(qubits[3]))

    elif env == 'CartPole-v1' and num_qubits == 4:
        measurement.append(cirq.Z(qubits[0]) * cirq.Z(qubits[1]))
        measurement.append(cirq.Z(qubits[2]) * cirq.Z(qubits[3]))

    elif num_qubits == 3 or num_qubits == 12:
        measurement.append(cirq.Z(qubits[0]))
        measurement.append(cirq.Z(qubits[1]))
        measurement.append(cirq.Z(qubits[2]))
    else:
        raise ValueError(f"Measurement not defined for env {env} and num_qubits {num_qubits}.")

    return measurement
