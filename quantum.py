import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
import secrets  # for cryptographically secure CRNG
import time
from scipy import stats
from scipy.stats import entropy


class RandomNumberGenerator:
    @staticmethod
    def classical_random(size=1000):
        """Generate random numbers using a cryptographically secure method."""
        return np.array([secrets.randbelow(2**32) for _ in range(size)], dtype=np.uint32)
    
    @staticmethod
    def quantum_random(size=1000):
        """Generate random numbers using a quantum circuit."""
        num_bits = 32  # Generate 32-bit numbers
        qc = QuantumCircuit(num_bits, num_bits)

        # Apply Hadamard gates for superposition
        for qubit in range(num_bits):
            qc.h(qubit)

        # Measure all qubits
        qc.measure(range(num_bits), range(num_bits))

        # Execute on Qiskit Aer simulator
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=size)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert bitstrings to integers
        random_numbers = np.array(
            [int(bitstring, 2) for bitstring in counts.keys() for _ in range(counts[bitstring])], dtype=np.uint32
        )
        return random_numbers[:size]  # Trim excess numbers


def analyze_randomness(numbers, name):
    """Perform statistical analysis on random numbers."""
    start_time = time.time()

    # Statistical metrics
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    calculated_entropy = calculate_entropy(numbers)

    # Uniformity test (Chi-squared)
    histogram, _ = np.histogram(numbers, bins=10)
    _, p_value = stats.chisquare(histogram)

    generation_time = time.time() - start_time

    return {
        'method': name,
        'mean': mean,
        'std_dev': std_dev,
        'entropy': calculated_entropy,
        'generation_time': generation_time,
        'uniformity_p_value': p_value,
    }


def calculate_entropy(numbers):
    """Calculate entropy of random numbers."""
    unique, counts = np.unique(numbers, return_counts=True)
    probabilities = counts / len(numbers)
    return entropy(probabilities)


def visualize_comparison(crng_stats, qrng_stats):
    """Visualize and compare statistical metrics."""
    metrics = ['mean', 'std_dev', 'entropy', 'generation_time']
    crng_values = [crng_stats[metric] for metric in metrics]
    qrng_values = [qrng_stats[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, crng_values, width, label='Classical RNG', color='blue', alpha=0.7)
    plt.bar(x + width / 2, qrng_values, width, label='Quantum RNG', color='red', alpha=0.7)

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison of Classical vs Quantum RNG')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Define sample size
    size = 10000

    # Generate random numbers
    crng_numbers = RandomNumberGenerator.classical_random(size=size)
    qrng_numbers = RandomNumberGenerator.quantum_random(size=size)

    # Analyze randomness
    crng_stats = analyze_randomness(crng_numbers, 'Classical RNG')
    qrng_stats = analyze_randomness(qrng_numbers, 'Quantum RNG')

    # Display statistics
    print("Classical RNG Statistics:")
    for key, value in crng_stats.items():
        print(f"{key}: {value:.4f}")

    print("\nQuantum RNG Statistics:")
    for key, value in qrng_stats.items():
        print(f"{key}: {value:.4f}")

    # Visualize comparison
    visualize_comparison(crng_stats, qrng_stats)


if _name_ == "_main_":
    main()




