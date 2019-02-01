#include "multilayerperceptron.h"
#include <iostream>
MultiLayerPerceptron::MultiLayerPerceptron(int inputDimension, int hiddenLayerSize, int outputDimension,
                                           double *inputs, double *outputs, double lr, double error, int sampleSize)
{
    this->inputDimension = inputDimension;
    this->hiddenLayerSize = hiddenLayerSize;
    this->outputDimension = outputDimension;
    this->inputs = inputs;
    this->outputs = outputs;
    this->lr = lr;
    this->error = error;
    this->sampleSize = sampleSize;
    howManyCycle = 0;
    weightsV = new double[inputDimension*hiddenLayerSize];
    weightW =new double[(hiddenLayerSize+1)*outputDimension];
}

MultiLayerPerceptron::~MultiLayerPerceptron()
{
    delete[] weightsV;
    delete[] weightW;
}

void MultiLayerPerceptron::randomWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    for (int i=0;i<inputDimension*hiddenLayerSize;i++)
    {
        weightsV[i] = dis(gen);
    }
    for (int i=0;i<(hiddenLayerSize+1)*outputDimension;i++)
    {
        weightW[i] = dis(gen);
    }

}

void MultiLayerPerceptron::learn()
{
    double e = 11000;
    while(e>error)
    {
//        std::cout << "error: " << e << "\n";
        e = 0;
        howManyCycle++;
        for(int i=0;i<sampleSize;i++)
        {
            // forward calculation start
            double netj[hiddenLayerSize];
            double netk[outputDimension];
            double fnetj[hiddenLayerSize+1];
            double fnetk[outputDimension];
            matrixMultiplication(weightsV, inputs+inputDimension*i, netj, hiddenLayerSize, inputDimension, 1);
            for(int j=0;j<hiddenLayerSize;j++) fnetj[j] = tanhf(netj[j]);
            fnetj[hiddenLayerSize] = -1;   // for bias
            matrixMultiplication(weightW, fnetj, netk, outputDimension, hiddenLayerSize+1, 1);
            for(int j=0;j<outputDimension;j++) fnetk[j] = tanhf(netk[j]);
            // forward calculation complate

            for(int j=0;j<outputDimension;j++) e += (outputs[i*outputDimension+j]-fnetk[j])*(outputs[i*outputDimension+j]-fnetk[j])*0.5;    // error calculation complate

            // backward calculation start
            for(int j=0;j<outputDimension;j++)
            {
                double constant = lr*(outputs[i*outputDimension+j]-fnetk[j])*0.5*(1-fnetk[j]*fnetk[j]);
                for(int k=0;k<hiddenLayerSize+1;k++)
                    weightW[j*(hiddenLayerSize+1)+k] += constant*fnetj[k];
            }

            for(int j=0;j<hiddenLayerSize;j++)
            {
                for(int f=0;f<inputDimension;f++)
                {
                    double constant = lr*0.5*(1-fnetj[j]*fnetj[j]);
                    double sum = 0;
                    for(int k=0;k<outputDimension;k++)
                        sum += 0.5*(outputs[i*outputDimension+k]-fnetk[k])*(1-fnetk[k]*fnetk[k])*weightW[k*(hiddenLayerSize+1)+j];
                    weightsV[j*inputDimension+f] += constant*sum*inputs[i*inputDimension+f];
                }
            }
            // backward calculation end

        }
    }

}

void MultiLayerPerceptron::learnWithMoment()
{
    double *deltaV = new double[inputDimension*hiddenLayerSize]();
    double *deltaW =new double[(hiddenLayerSize+1)*outputDimension]();
    double e = 11000;
    while(e>error)
    {
        e = 0;
        howManyCycle++;
        for(int i=0;i<sampleSize;i++)
        {
            // forward calculation start
            double netj[hiddenLayerSize];
            double netk[outputDimension];
            double fnetj[hiddenLayerSize+1];
            double fnetk[outputDimension];
            matrixMultiplication(weightsV, inputs+inputDimension*i, netj, hiddenLayerSize, inputDimension, 1);
            for(int j=0;j<hiddenLayerSize;j++) fnetj[j] = tanhf(netj[j]);
            fnetj[hiddenLayerSize] = -1;   // for bias
            matrixMultiplication(weightW, fnetj, netk, outputDimension, hiddenLayerSize+1, 1);
            for(int j=0;j<outputDimension;j++) fnetk[j] = tanhf(netk[j]);
            // forward calculation complate

            for(int j=0;j<outputDimension;j++) e += (outputs[i*outputDimension+j]-fnetk[j])*(outputs[i*outputDimension+j]-fnetk[j])*0.5;    // error calculation complate

            // backward calculation start
            for(int j=0;j<outputDimension;j++)
            {
                double constant = lr*(outputs[i*outputDimension+j]-fnetk[j])*0.5*(1-fnetk[j]*fnetk[j]);
                for(int k=0;k<hiddenLayerSize+1;k++)
                {
                    weightW[j*(hiddenLayerSize+1)+k] += constant*fnetj[k] + 0.8*deltaW[j*(hiddenLayerSize+1)+k];
                    deltaW[j*(hiddenLayerSize+1)+k] = constant*fnetj[k];
                }
            }

            for(int j=0;j<hiddenLayerSize;j++)
            {
                for(int f=0;f<inputDimension;f++)
                {
                    double constant = lr*0.5*(1-fnetj[j]*fnetj[j]);
                    double sum = 0;
                    for(int k=0;k<outputDimension;k++)
                        sum += 0.5*(outputs[i*outputDimension+k]-fnetk[k])*(1-fnetk[k]*fnetk[k])*weightW[k*(hiddenLayerSize+1)+j];
                    weightsV[j*inputDimension+f] += constant*sum*inputs[i*inputDimension+f] + 0.8*deltaV[j*inputDimension+f];
                    deltaV[j*inputDimension+f] = constant*sum*inputs[i*inputDimension+f];
                }
            }
            // backward calculation end

        }
    }
    delete[] deltaV;
    delete[] deltaW;
}

void MultiLayerPerceptron::setWeightsV(double *value)
{
    for(int i=0;i<inputDimension*hiddenLayerSize;i++)
        weightsV[i] = value[i];
}

void MultiLayerPerceptron::setWeightW(double *value)
{
    for(int i=0;i<(hiddenLayerSize+1)*outputDimension;i++)
        weightW[i] = value[i];
}

void MultiLayerPerceptron::copyWeightsV(double *value)
{
    for(int i=0;i<inputDimension*hiddenLayerSize;i++)
        value[i] = weightsV[i];
}

void MultiLayerPerceptron::copyWeightW(double *value)
{
    for(int i=0;i<(hiddenLayerSize+1)*outputDimension;i++)
        value[i] = weightW[i];
}

void MultiLayerPerceptron::setInputs(double *value)
{
    inputs = value;
}

void MultiLayerPerceptron::setOutputs(double *value)
{
    outputs = value;
}

void MultiLayerPerceptron::setLr(double value)
{
    lr = value;
}

void MultiLayerPerceptron::setError(double value)
{
    error = value;
}

int MultiLayerPerceptron::getHowManyCycle() const
{
    return howManyCycle;
}

void MultiLayerPerceptron::setSampleSize(int value)
{
    sampleSize = value;
}

void MultiLayerPerceptron::setHiddenLayerSize(int value)
{
    hiddenLayerSize = value;
}

void MultiLayerPerceptron::setOutputDimension(int value)
{
    outputDimension = value;
}

void MultiLayerPerceptron::setInputDimension(int value)
{
    inputDimension = value;
}

double *MultiLayerPerceptron::test(double *inp, double *out)
{
    // forward calculation start
    double netj[hiddenLayerSize];
    double netk[outputDimension];
    double fnetj[hiddenLayerSize+1];
    double fnetk[outputDimension];
    matrixMultiplication(weightsV, inp, netj, hiddenLayerSize, inputDimension, 1);
    for(int j=0;j<hiddenLayerSize;j++) fnetj[j] = tanhf(netj[j]);
    fnetj[hiddenLayerSize] = -1;   // for bias
    matrixMultiplication(weightW, fnetj, netk, outputDimension, hiddenLayerSize+1, 1);
    for(int j=0;j<outputDimension;j++) fnetk[j] = tanhf(netk[j]);
    // forward calculation complate
    for(int i=0;i<outputDimension;i++) out[i] = fnetk[i];
}

double *MultiLayerPerceptron::getWeightsV() const
{
    return weightsV;
}

double *MultiLayerPerceptron::getWeightW() const
{
    return weightW;
}

void MultiLayerPerceptron::matrixMultiplication(double *a, double *b, double *c, int r1, int c1, int c2)
{
    // r1 number of rows of a matrix
    // c1 number of columns of a matrix
    // c2 number of columns of b matrix
    for(int i=0; i<r1; ++i)
    {
        for(int j=0; j<c2; ++j)
        {
            c[i*c2+j] = 0;
            for(int k=0; k<c1; ++k)
            {
                c[i*c2+j]+=a[i*c1+k]*b[k*c2+j];
            }
        }
    }
}
