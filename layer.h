#ifndef LAYER_H
#define LAYER_H

#include <neuron.h>
#include <vector>

/**
 * @brief Layer class
 *
 */
class Layer
{
public:
    /**
     * @brief Default Constructer
     *
     */
    Layer();
    /**
     * @brief Destructor
     *
     */
    ~Layer();
    /**
     * @brief Constructer
     *
     * @param classSize number of class
     * @param lr learning rate
     * @param inp input vector
     * @param out output vector, {1,1,2,3}
     * 1,2 and 3 are different classes
     * @param error error rate
     * @param inpDim dimension of inputs
     * @param sampleSize number of samples
     */
    Layer(int classSize, double lr, double *inp, double *out, double error, int inpDim, int sampleSize);
    /**
     * @brief Convert output vector to format of neuron class
     *
     */
    void convertOutputs();
    /**
     * @brief set neurons
     *
     */
    void setNeurons();
    /**
     * @brief train with discrete function
     *
     */
    void trainPerceptron();
    /**
     * @brief train with continuous function
     *
     */
    void trainDelta();
    /**
     * @brief returns pointer of weights
     *
     * @return double
     */
    double *getWeights();
    /**
     * @brief returns number of cycles in training
     *
     * @return unsigned int
     */
    unsigned int getHowManyCycle();
private:
    int *colorsNum; /**< TODO: describe */
    unsigned int cycle; /**< TODO: describe */
    int inputDimension, sampleSize, classSize; /**< TODO: describe */
    double *weights, *inputs, *outputs, lr, error; /**< TODO: describe */
    double *transformedoutputs; /**< TODO: describe */
    Neuron **neurons; /**< TODO: describe */
};

#endif // LAYER_H
