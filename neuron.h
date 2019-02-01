#ifndef NORON_H
#define NORON_H

#include <random>
#include <math.h>
#include <iostream>
/**
 * @brief Neuron Class
 *
 */
class Neuron
{
public:
    /**
     * @brief Default constructer
     *
     */
    Neuron();
    /**
     * @brief Create neuron instance and
     * set following parameters
     *
     * @param w weigth vector
     * @param lr learning rate
     * @param inp input vector
     * @param out output vector, {1,1,-1}
     * 1 and -1 are different classes
     * @param error error rate
     * @param inpDim dimension of inputs
     * @param sampleSize number of samples
     */
    Neuron(double *w, double lr, double *inp, double *out, double error, int inpDim, int sampleSize);
    /**
     * @brief Destructor
     *
     */
    ~Neuron();
    /**
     * @brief Set input dimension and
     * number of samples
     *
     * @param dim input dimesion
     * @param sampleSize number of samples
     */
    void Configure(int dim, int sampleSize);
    /**
     * @brief set weights
     *
     * @param w weight vector
     */
    void setWeights(double *w);
    /**
     * @brief set weights with random values
     *
     */
    void randomWeights();
    /**
     * @brief set input vector
     *
     * @param value pointer of input vector
     */
    void setInputs(double *value);
    /**
     * @brief returns output vector
     *
     * @return double
     */
    double *getOutputs() const;
    /**
     * @brief set output vector
     *
     * @param value pointer of output vector
     */
    void setOutputs(double *value);
    /**
     * @brief returns learning rate
     *
     * @return double
     */
    double getLr() const;
    /**
     * @brief set learning rate
     *
     * @param value
     */
    void setLr(double value);
    /**
     * @brief returns current weights
     *
     * @return double pointer of weight array
     */
    double *getWeights() const;
    /**
     * @brief train neuron with discrete
     * function, The function is bipolar
     * signum
     *
     */
    void perceptronRule();
    /**
     * @brief train neuron with continuous
     * function, The function is bipolar
     * sigmoid
     *
     */
    void deltaRule();
    /**
     * @brief returns error rate
     *
     * @return double
     */
    double getError() const;
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

private:
    int inputDimension, sampleSize, howManyCycle; /**< TODO: describe */
    double *weights, *inputs, *outputs, lr, error; /**< TODO: describe */
    /**
     * @brief calculate output with discrete
     * function and returns it
     *
     * @param order order of input value, {inp1, inp2, inp3}
     * inp1, inp2, inp3 are vectors which size is inputSize,
     * if order is 2 than calculate output of inp2
     * @return int
     */
    int calcOutput(int order);
    /**
     * @brief calculate output with continuous
     * function and returns it
     *
     * @param order order of input value, {inp1, inp2, inp3}
     * inp1, inp2, inp3 are vectors which size is inputSize,
     * if order is 2 than calculate output of inp2
     * @return double
     */
    double calcOutputActivation(int order);
    /**
     * @brief update weights
     *
     * @param order
     * @param output
     */
    void updateWeights(int order, int output);
    /**
     * @brief update weights with continuous function
     *
     * @param order
     * @param output
     */
    void updateWeightsActivation(int order, double output);

};

#endif // NORON_H
