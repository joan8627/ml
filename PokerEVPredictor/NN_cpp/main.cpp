#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace std;

// Will be used for neurons, inputs, outputs
typedef vector<double> Vector;

// Used for Weights between layers
typedef vector<Vector> Matrix;

// Radnom number gen between [-0.5, 0.5]
// To generate initial weights and biases
double randomWeight() {
    return ((double) rand() / RAND_MAX) - 0.5;
}

// Initialize weight matrix with random weights
Matrix createMatrix(int i, int j) {
    Matrix m(i, Vector(j));
    for (int r = 0; r < i; r++) {
        for (int s = 0; s < j; s++) {
            m[r][s] = randomWeight();
        }
    }
    return m;
}

/* todo - use HE
 * Matrix createMatrixHe(int rows, int fan_in) {
     static std::mt19937 gen(42);
     double stddev = std::sqrt(2.0 / fan_in);
     std::normal_distribution<> dis(0.0, stddev);

     Matrix m(rows, Vector(fan_in));
     for (int r = 0; r < rows; r++) {
         for (int c = 0; c < fan_in; c++) {
             m[r][c] = dis(gen);
         }
     }
     return m;
 }
 */
// Initialize biases with random weights
Vector createVector(int size) {
    Vector v(size);
    for (int r = 0; r < size; r++) {
        v[r] = randomWeight();
    }
    return v;
}

// Activation function (Rectified Linear Unit)
// todo: maybe try Leaky ReLU
double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

//
double reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double softplus(double x) {
    return log(1.0 + exp(x));
}

double softplusDerivative(double x) {
    return 1.0 / (1.0 + exp(-x));
}

struct Network {
    // Hidden layer 1
    Matrix weightsInputToHidden1;
    Vector biasHidden1;

    // Hidden layer 2
    Matrix weightsHidden1ToHidden2;
    Vector biasHidden2;

    // Output heads
    Matrix weightsHidden2ToOutputQuantiles;
    Vector biasOutputQuantiles;
};

void forward(const Network& net,
    const Vector& input,
    Vector& hiddenLayer1Activations,
    Vector& hiddenLayer2Activations,
    Vector& quantiles
) {

    hiddenLayer1Activations = Vector(net.weightsInputToHidden1.size());
    for (int h1 = 0; h1 < net.weightsInputToHidden1.size(); h1++) {
        double sum = net.biasHidden1[h1];
        for (int in = 0; in < input.size(); in++) {
            sum += net.weightsInputToHidden1[h1][in] * input[in];
        }
        hiddenLayer1Activations[h1] = relu(sum);
    }

    hiddenLayer2Activations = Vector(net.weightsHidden1ToHidden2.size());
    for (int h2 = 0; h2 < net.weightsHidden1ToHidden2.size(); h2++) {
        double sum = net.biasHidden2[h2];
        for (int h1 = 0; h1 < hiddenLayer1Activations.size(); h1++) {
            sum += net.weightsHidden1ToHidden2[h2][h1] * hiddenLayer1Activations[h1];
        }
        hiddenLayer2Activations[h2] = relu(sum);
    }

    quantiles = Vector(net.biasOutputQuantiles.size());
    for (int n = 0; n < net.weightsHidden2ToOutputQuantiles[0].size(); n++) {
        double sum_current_quantile = net.biasOutputQuantiles[n];
        for (int m = 0; m < hiddenLayer2Activations.size(); m++) {
            sum_current_quantile += net.weightsHidden2ToOutputQuantiles[m][n] * hiddenLayer2Activations[m];
        }
        quantiles[n] = sum_current_quantile;
    }
}

double quantileLoss(
    const Vector& quantiles,
    const Vector& taus,
    double y
) {
    double loss = 0.0;
    for (int i = 0; i < quantiles.size(); i++) {
        double q = quantiles[i];
        double tau = taus[i];
        double error = y - q;

        if (error > 0) {
            loss += tau * error;
        } else {
            loss += (tau - 1.0) * error;
        }
    }
    return loss;
}

Vector quantileLossGradient(const Vector& quantiles,
    const Vector& taus,
    double y) {
        // Will store output here
        Vector grad(quantiles.size());

        for (int i = 0; i < quantiles.size(); i++) {
            double q = quantiles[i];
            double tau = taus[i];

            if (y > q) {
                grad[i] = -tau;
            } else {
                grad[i] = -(tau - 1.0);
            }
        }
        return grad;
    }

/*

    void trainBatch(
        Network& net,
        const vector<Vector>& inputs,
        const vector<double>& ys,
        const Vector& taus,
        double learningRate
    ) {
        int batchSize = inputs.size();

        // Accumulators for gradients (initialized to zero)
        Matrix gradW1 = createMatrix(hidden1Size, inputSize);       // zeros
        Vector gradB1 = createVector(hidden1Size);                  // zeros
        Matrix gradW2 = createMatrix(hidden2Size, hidden1Size);     // zeros
        Vector gradB2 = createVector(hidden2Size);                  // zeros
        Matrix gradW3 = createMatrix(hidden2Size, numQuantiles);    // zeros
        Vector gradB3 = createVector(numQuantiles);                // zeros

        // Accumulate gradients over all samples in batch
        for (int sample = 0; sample < batchSize; sample++) {
            Vector hiddenLayer1Activations, hiddenLayer2Activations, outputQuantiles;
            forward(net, inputs[sample], hiddenLayer1Activations, hiddenLayer2Activations, outputQuantiles);

            Vector dLoss_dQuantiles = quantileLossGradient(outputQuantiles, taus, ys[sample]);

            // Backprop hidden2
            Vector dLoss_dHidden2(hidden2Size, 0.0);
            for (int h = 0; h < hidden2Size; h++) {
                for (int q = 0; q < numQuantiles; q++) {
                    dLoss_dHidden2[h] += dLoss_dQuantiles[q] * net.weightsHidden2ToOutputQuantiles[h][q];
                }
                if (hiddenLayer2Activations[h] <= 0.0) dLoss_dHidden2[h] = 0.0;
            }

            // Backprop hidden1
            Vector dLoss_dHidden1(hidden1Size, 0.0);
            for (int h1 = 0; h1 < hidden1Size; h1++) {
                for (int h2 = 0; h2 < hidden2Size; h2++) {
                    dLoss_dHidden1[h1] += dLoss_dHidden2[h2] * net.weightsHidden1ToHidden2[h2][h1];
                }
                if (hiddenLayer1Activations[h1] <= 0.0) dLoss_dHidden1[h1] = 0.0;
            }

            // Accumulate weight gradients
            for (int h2 = 0; h2 < hidden2Size; h2++) {
                for (int q = 0; q < numQuantiles; q++) {
                    gradW3[h2][q] += dLoss_dQuantiles[q] * hiddenLayer2Activations[h2];
                }
            }
            for (int q = 0; q < numQuantiles; q++) {
                gradB3[q] += dLoss_dQuantiles[q];
            }

            for (int h2 = 0; h2 < hidden2Size; h2++) {
                for (int h1 = 0; h1 < hidden1Size; h1++) {
                    gradW2[h2][h1] += dLoss_dHidden2[h2] * hiddenLayer1Activations[h1];
                }
                gradB2[h2] += dLoss_dHidden2[h2];
            }

            for (int h1 = 0; h1 < hidden1Size; h1++) {
                for (int in = 0; in < inputSize; in++) {
                    gradW1[h1][in] += dLoss_dHidden1[h1] * inputs[sample][in];
                }
                gradB1[h1] += dLoss_dHidden1[h1];
            }
        }

        // Apply averaged gradients
        for (int h2 = 0; h2 < hidden2Size; h2++) {
            for (int q = 0; q < numQuantiles; q++) {
                net.weightsHidden2ToOutputQuantiles[h2][q] -= learningRate * gradW3[h2][q] / batchSize;
            }
        }
        // ... same pattern for all other weights/biases
    }

 */
void trainOneSample(
    Network& net,
    const Vector& input,
    double y,
    const Vector& taus,
    double learningRate
) {
    Vector hiddenLayer1Activations;
    Vector hiddenLayer2Activations;
    Vector outputQuantiles;

    forward(net, input, hiddenLayer1Activations, hiddenLayer2Activations, outputQuantiles);

    Vector dLoss_dQuantiles = quantileLossGradient(outputQuantiles, taus, y);

    // -----------------------------
    // Backprop into hidden2
    // -----------------------------
    Vector dLoss_dHidden2(hiddenLayer2Activations.size(), 0.0);
    for (int h = 0; h < hiddenLayer2Activations.size(); h++) {
        for (int q = 0; q < outputQuantiles.size(); q++) {
            dLoss_dHidden2[h] += dLoss_dQuantiles[q] * net.weightsHidden2ToOutputQuantiles[h][q];
        }
        // relu
        if (hiddenLayer2Activations[h] <= 0.0) {
            dLoss_dHidden2[h] = 0.0;
        }
    }

    // -----------------------------
    // Backprop into hidden1
    // -----------------------------
    Vector dLoss_dHidden1(hiddenLayer1Activations.size(), 0.0);
    for (int h1 = 0; h1 < hiddenLayer1Activations.size(); h1++) {
        for (int h2 = 0; h2 < hiddenLayer2Activations.size(); h2++) {
            dLoss_dHidden1[h1] += dLoss_dHidden2[h2] * net.weightsHidden1ToHidden2[h2][h1];
            // !!! pay attention to indices - not * net.weightsHidden1ToHidden2[h1][h2]
        }
        // relu
        if (hiddenLayer1Activations[h1] <= 0.0) {
            dLoss_dHidden1[h1] = 0.0;
        }
    }

     // -----------------------------
     // Update output quantile layer
     // -----------------------------
     for (int h2 = 0; h2 < hiddenLayer2Activations.size(); h2++) {
         for (int q = 0; q < outputQuantiles.size(); q++) {
             net.weightsHidden2ToOutputQuantiles[h2][q] -= learningRate * dLoss_dQuantiles[q] * hiddenLayer2Activations[h2];
         }
     }
     for (int q = 0; q < outputQuantiles.size(); q++) {
         net.biasOutputQuantiles[q] -= learningRate * dLoss_dQuantiles[q];
     }

     // -----------------------------
     // Update hidden1 -> hidden2 layer
     // -----------------------------
     for (int h2 = 0; h2 < hiddenLayer2Activations.size(); h2++) {
        for (int h1 = 0; h1 < hiddenLayer1Activations.size(); h1++) {
            net.weightsHidden1ToHidden2[h2][h1] -= learningRate * dLoss_dHidden2[h2] * hiddenLayer1Activations[h1];
        }
        net.biasHidden2[h2] -= learningRate * dLoss_dHidden2[h2];
     }

     // -----------------------------
     // Update input -> hidden1 layer
     // -----------------------------
     for (int h1 = 0; h1 < hiddenLayer1Activations.size(); h1++) {
         for (int in = 0; in < input.size(); in++) {
             net.weightsInputToHidden1[h1][in] -= learningRate * dLoss_dHidden1[h1] * input[in];
         }
         net.biasHidden1[h1] -= learningRate * dLoss_dHidden1[h1];
     }
}


int main() {
    srand(42);

    int inputSize = 5;
    int hidden1Size = 8;
    int hidden2Size = 8;

    Vector taus = {0.1, 0.5, 0.9};
    int numQuantiles = taus.size();

    Network net;

    net.weightsInputToHidden1 = createMatrix(hidden1Size, inputSize);
    net.biasHidden1 = createVector(hidden1Size);

    net.weightsHidden1ToHidden2 = createMatrix(hidden2Size, hidden1Size);
    net.biasHidden2 = createVector(hidden2Size);

    net.weightsHidden2ToOutputQuantiles = createMatrix(hidden2Size, numQuantiles);
    net.biasOutputQuantiles = createVector(numQuantiles);

    vector<Vector> trainingInputs = {
        {0.8, 0.2, 0.5, 0.7, 0.1},
        {0.1, 0.9, 0.3, 0.2, 0.8},
        {0.6, 0.4, 0.9, 0.5, 0.3}
    };

    Vector trainingOutcomes = {
        120.0,
        -80.0,
        40.0
    };

    double learningRate = 0.001;
    int totalEpoch = 10; //10000;

    /*
     * int batchSize = 3;  // or smaller if you have more data
     for (int epoch = 0; epoch < totalEpoch; epoch++) {
         trainBatch(net, trainingInputs, trainingOutcomes, taus, learningRate);
     }
     */
    for (int epoch = 0; epoch < totalEpoch; epoch++) {
        for (int i = 0; i < trainingInputs.size(); i++) {
            trainOneSample(
                net,
                trainingInputs[i],
                trainingOutcomes[i],
                taus,
                learningRate
            );
        }
    }

    /////
    /// test
    ///

    Vector hidden1;
    Vector hidden2;
    Vector predictedQuantiles;

    Vector testInput = {0.7, 0.3, 0.6, 0.8, 0.2};

    forward(net, testInput, hidden1, hidden2, predictedQuantiles);

    cout << "Predicted quantiles:" << endl;

    for (int i = 0; i < predictedQuantiles.size(); i++) {
        cout << "q" << taus[i] << " = "
                << predictedQuantiles[i] << endl;
    }

    return 0;

}
