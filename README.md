# NeuralNetwork
Neural Network Implementation with C++ 

You can see how to implement neural networks.

A Noron Example
    
    // inputs length is 9 because of input dimension * number of sample
    double inputs[9] = {-2,-2,-1,0,-2,-1,3,5,-1}; // -1 bias 
    double outputs[3] = {-1,-1,1};
    
    Neuron neuron;
    neuron.Configure(3, items.size());
    neuron.randomWeights();
    neuron.setInputs(inputs);
    neuron.setOutputs(outputs);
    neuron.setLr(0.5);
    neuron.setError(0.01);
    neuron.deltaRule();  //train neuron
    // neuron.perceptronRule()
    
    double *w = noron.getWeights(); 
    
A Layer Example

    double inputs[9] = {-2,-2,-1,0,-2,-1,3,5,-1}; // -1 bias 
    double outputs[3] = {1,2,3};
    
    Layer layer(3, 0.5, inputs, outputs, 0.01, 3, 3);
    layer.convertOutputs();
    layer.setNeurons();
    layer.trainDelta();
    // layer.trainPerceptron();
    
    double *w = layer.getWeights();
    
Multi Layer Perceptron Example(Exor)

    double inputs[12] = {-5,-5,-1,5,-5,-1,-5,5,-1,5,5,-1}; // -1 bias 
    double outputs[4] = {-1,1,1,-1}; 
    
    MultiLayerPerceptron p(3,4,1,inputs,outputs,0.5,0.1,4);
    p.randomWeights();
    p.learn();
