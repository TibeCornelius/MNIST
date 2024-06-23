namespace Ai.MNIST.NeuralNetworks
{
    public readonly struct NetworkJsonFormat( int layercount, int[] neuronCount, double[,] wheights, double[] biases )
    {
        private readonly int LayerCount = layercount ; 
        private readonly int[] NeuronCount = neuronCount;
        private readonly double[,] Wheights = wheights;
        private readonly double[] Biasese = biases;
    }
}