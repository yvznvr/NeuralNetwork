#include "layer.h"

Layer::Layer()
{
    inputDimension = 0;
    sampleSize = classSize = 0;
    lr = error = cycle = 0;
    transformedoutputs = nullptr;
    weights = nullptr;
    inputs = nullptr;
    outputs = nullptr;
    neurons = nullptr;
    colorsNum = nullptr;
}

Layer::~Layer()
{
    delete[] transformedoutputs;
    delete[] weights;
    delete[] colorsNum;
    for(int i=0; i<classSize;i++) delete neurons[i];
    delete[] neurons;
}

Layer::Layer(int classSize, double lr, double *inp, double *out, double error, int inpDim, int sampleSize)
{
    neurons = new Neuron*[classSize];
    colorsNum = new int[classSize];
    inputs = inp;
    outputs = out;
    this->lr = lr;
    this->error = error;
    inputDimension = inpDim;
    this->sampleSize = sampleSize;
    this->classSize = classSize;
    for(int i=0; i<classSize;i++) neurons[i] = new Neuron();
}

void Layer::convertOutputs()
{
    // convert {1,2,3} to {1,0,0,0,1,0,0,0,1}
    // first 3 elements belong to first neuron
    // last 3 elements belong to third neuron
    transformedoutputs = new double[classSize*sampleSize];
    for(int i=0;i<classSize;i++)
    {
        for(int j=0;j<sampleSize;j++)
        {
            if(i+1==outputs[j]) transformedoutputs[sampleSize*i+j] = 1;
            else transformedoutputs[sampleSize*i+j] = -1;
        }
    }
}

void Layer::setNeurons()
{
    // set neurons
    for(int i=0; i<classSize;i++)
    {
        neurons[i]->Configure(inputDimension,sampleSize);
        neurons[i]->randomWeights();
        neurons[i]->setInputs(inputs);
        neurons[i]->setOutputs(transformedoutputs+i*(sampleSize));
        neurons[i]->setLr(lr);
        neurons[i]->setError(error);
    }

}

void Layer::trainPerceptron()
{
    for(int i=0;i<classSize;i++)
        neurons[i]->perceptronRule();
}

void Layer::trainDelta()
{
    for(int i=0;i<classSize;i++)
        neurons[i]->deltaRule();
}

double *Layer::getWeights()
{
    weights = new double[classSize*inputDimension];
    for(int i=0;i<classSize;i++)
    {
        double *temp = neurons[i]->getWeights();
        for(int j=0;j<inputDimension;j++)
        {
            weights[i*inputDimension+j] = temp[j];
        }
    }
    return weights;
}


unsigned int Layer::getHowManyCycle()
{
    cycle = 0;
    for (int i=0; i<classSize; i++)
    {
        unsigned int cy = neurons[i]->getHowManyCycle();
        if(cy>cycle) cycle = cy;
    }
    return cycle;
}

