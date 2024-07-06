namespace MNIST.NeuralNetworks
{
    public struct StNeuron
    {
        public double input;
        public double output;
        public double biases;
        public double biasesGradient;

    }
    public ref struct RefStNeuron
    {
        public ref StNeuron[] Neurons;
        public RefStNeuron( ref StNeuron[] neurons )
        {
            this.Neurons = ref neurons;
        }
    }
    

}