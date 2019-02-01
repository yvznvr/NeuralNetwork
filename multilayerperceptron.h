#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <random>

/**
 * @brief Multilayer perceptron class
 * for 2 layer neural networks
 *
 */
class MultiLayerPerceptron
{
public:
    /**
     * @brief Constructer
     *
     * @param inputDimension dimension of input
     * @param hiddenLayerSize number of neurons for hidden layer
     * @param outputDimension dimension of output, should be number of class
     * @param inputs input vector
     * @param outputs ouutput vector, must be {1,-1,-1,...} for 3 class. Only
     * one neuron's output can be 1.
     * @param lr learning rate
     * @param error error rate
     * @param sampleSize number of samples
     */
    MultiLayerPerceptron(int inputDimension, int hiddenLayerSize, int outputDimension,
                         double *inputs, double *outputs, double lr, double error, int sampleSize);
    /**
     * @brief Destructor
     *
     */
    ~MultiLayerPerceptron();
    /**
     * @brief set weights with randum values
     *
     */
    void randomWeights();
    /**
     * @brief trains network
     *
     */
    void learn();
    /**
     * @brief trains network wit moment
     *
     */
    void learnWithMoment();
    /**
     * @brief sets weight vector between input layer and hidden layer
     *
     * @param value
     */
    void setWeightsV(double *value);
    /**
     * @brief sets weight vector between hidden layer and output layer
     *
     * @param value
     */
    void setWeightW(double *value);
    /**
     * @brief copies weight vector between input layer and hidden layer
     *
     * @param value
     */
    void copyWeightsV(double *value);
    /**
     * @brief copies weight vector between hidden layer and output layer
     *
     * @param value
     */
    void copyWeightW(double *value);
    /**
     * @brief sets input vector
     *
     * @param value
     */
    void setInputs(double *value);
    /**
     * @brief sets output vector
     *
     * @param value
     */
    void setOutputs(double *value);
    /**
     * @brief set learning rate
     *
     * @param value
     */
    void setLr(double value);
    /**
     * @brief set error rate
     *
     * @param value
     */
    void setError(double value);
    /**
     * @brief returns number of cycles in training
     *
     * @return int
     */
    int getHowManyCycle() const;
    /**
     * @brief sets number of test sample
     *
     * @param value
     */
    void setSampleSize(int value);
    /**
     * @brief sets neuron size of hidden layer
     *
     * @param value
     */
    void setHiddenLayerSize(int value);
    /**
     * @brief sets output dimension, shoul be
     * class size
     *
     * @param value
     */
    void setOutputDimension(int value);
    /**
     * @brief sets dimension of input
     *
     * @param value
     */
    void setInputDimension(int value);
    /**
     * @brief tests data and returns
     * calculated output vector
     *
     * @param inp
     * @param out
     * @return double
     */
    double *test(double *inp, double *out);
    /**
     * @brief returns weight vector between input layer and hidden layer
     *
     * @return double
     */
    double *getWeightsV() const;
    /**
     * @brief returns weight vector between hidden layer and output layer
     *
     * @return double
     */
    double *getWeightW() const;

private:
    int inputDimension, outputDimension, hiddenLayerSize, sampleSize, howManyCycle;
    double *weightsV, *weightW, *inputs, *outputs, lr, error;
    double *calcActivation(int order);
    double *calcDerivativeActivation(int order);
    void matrixMultiplication(double *a, double *b, double *c, int r1, int c1, int c2);

};

#endif // MULTILAYERPERCEPTRON_H
