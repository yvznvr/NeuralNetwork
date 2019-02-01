#include "neuron.h"

Neuron::Neuron()
{
    inputDimension = 0;
    sampleSize = 0;
    lr = error = 0;
    weights = nullptr;
    inputs = nullptr;
    outputs = nullptr;
}

Neuron::Neuron(double *w, double lr, double *inp, double *out, double error, int inpDim, int sampleSize)
{
    inputs = inp;
    outputs = out;
    this->lr = lr;
    this->error = error;
    inputDimension = inpDim;
    this->sampleSize = sampleSize;
    setWeights(w);
}

Neuron::~Neuron()
{
    delete[] weights;
}

void Neuron::Configure(int dim, int sampleSize)
{
    inputDimension = dim;
    this->sampleSize = sampleSize;
}

void Neuron::setWeights(double *w)
{
    weights = new double[inputDimension];
    for (int i=0;i<inputDimension;i++)
    {
        weights[i] = w[i];
    }
}

void Neuron::randomWeights()
{
    weights = new double[inputDimension];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    for (int i=0;i<inputDimension;i++)
    {
        weights[i] = dis(gen);
    }

}

void Neuron::setInputs(double *value)
{
    inputs = value;
}

double *Neuron::getOutputs() const
{
    return outputs;
}

void Neuron::setOutputs(double *value)
{
    outputs = value;
}

double Neuron::getLr() const
{
    return lr;
}

void Neuron::setLr(double value)
{
    lr = value;
}

double *Neuron::getWeights() const
{
    return weights;
}

void Neuron::perceptronRule()
{
    int counter = 0;
    howManyCycle = 0;
    while(counter!=sampleSize)
    {
        counter = 0;
        for (int i=0;i<sampleSize;i++)
        {
            int o = calcOutput(i);
            if(o == outputs[i]) { counter++; continue; }
            updateWeights(i, o);
        }
        howManyCycle++;
    }
}

void Neuron::deltaRule()
{
    double e = 1000;
    howManyCycle = 0;
    while(0.5*e>error)
    {
        //std::cout << "error: " << e << std::endl;
        e = 0;
        for (int i=0;i<sampleSize;i++)
        {
            double o = calcOutputActivation(i);
            e += (outputs[i]-o)*(outputs[i]-o);
            updateWeightsActivation(i, o);
        }
        howManyCycle++;
    }
}

double Neuron::getError() const
{
    return error;
}

void Neuron::setError(double value)
{
    error = value;
}

int Neuron::getHowManyCycle() const
{
    return howManyCycle;
}


int Neuron::calcOutput(int order)
{
    double net = 0;
    for(int i=0;i<inputDimension;i++)
    {
        net += weights[i]*inputs[order*inputDimension+i];
    }
    if(net>0) return 1;
    return -1;
}

double Neuron::calcOutputActivation(int order)
{
    double net = 0;
    for(int i=0;i<inputDimension;i++)
    {
        net += weights[i]*(double)inputs[order*inputDimension+i];
    }
    return (2.0 / (exp(-net) + 1.0)) - 1.0;
//    return 1.0/(1.0 + exp(-net));
}

void Neuron::updateWeights(int order, int output)
{
    double constant = lr*(outputs[order]-output);
    for (int i=0; i<inputDimension; i++)
    {
        weights[i] = weights[i]+constant*inputs[inputDimension*order+i];
    }
}

void Neuron::updateWeightsActivation(int order, double output)
{
    double derivative = 0.5*(1 - output*output);
//    double derivative = output*(1-output);
    double constant = lr*(outputs[order]-output)*derivative;
    for (int i=0; i<inputDimension; i++)
    {
        weights[i] = weights[i]+constant*inputs[inputDimension*order+i];
    }
}
