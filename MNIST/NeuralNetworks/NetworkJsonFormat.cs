using Ai.MNIST.Util;

namespace Ai.MNIST.NeuralNetworks
{
    public class NetworkJsonFormat
    {
        public int LayerCount { get; }
        public int[] NeuronCount { get; }
        public double[][][] Weights { get; }
        public double[][] Biases { get; }


        public NetworkJsonFormat(int layerCount, int[] neuronCount, List<double[,]> weights, List<double[]> biases)
        {
            LayerCount = layerCount;
            NeuronCount = neuronCount;
            Weights = Converter.List2DArrayToJaggedArray( weights );
            Biases = Converter.ListToJaggedArray( biases );
        }

    }
}