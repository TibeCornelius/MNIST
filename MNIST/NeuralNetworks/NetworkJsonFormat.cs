using Ai.MNIST.Util;
using System.Text.Json.Serialization;
namespace Ai.MNIST.NeuralNetworks
{
    public class NetworkJsonFormat
    {
        public int LayerCount { get; set; }
        public int[] NeuronCount { get; set; }
        public double[][][] Weights { get; set; }
        public double[][] Biases { get; set; }


        public NetworkJsonFormat(int layerCount, int[] neuronCount, List<double[,]> weights, List<double[]> biases)
        {
            LayerCount = layerCount;
            NeuronCount = neuronCount;
            Weights = Converter.List2DArrayToJaggedArray( weights );
            Biases = Converter.ListToJaggedArray( biases );
        }
        [JsonConstructor]
        public NetworkJsonFormat(int layerCount, int[] neuronCount, double[][][] weights, double[][] biases)
        {
            LayerCount = layerCount;
            NeuronCount = neuronCount;
            Weights = weights;
            Biases = biases;
        }

    }
}