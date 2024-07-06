using MNIST.Util;
using System.Text.Json.Serialization;
namespace MNIST.NeuralNetworks
{
    public class NetworkJsonFormat
    {
        public int LayerCount { get; set; }
        public int[] NeuronCount { get; set; }
        public double[][][] Weights { get; set; }
        public double[][] Biases { get; set; }
        public int myActivationType { get; set; }
        public bool iamImageRecognizer { get; set; }
        public int myImageToRecognize { get; set; }

        // Parameterless constructor for deserialization
        public NetworkJsonFormat() { }

        public NetworkJsonFormat(int layerCount, int[] neuronCount, List<double[,]> weights, List<double[]> biases, ActivationFunctionOptions activationtype, bool iamImageRecognizer, int imageToRecognize)
        {
            this.LayerCount = layerCount;
            this.NeuronCount = neuronCount;
            this.Weights = Converter.List2DArrayToJaggedArray(weights);
            this.Biases = Converter.ListToJaggedArray(biases);
            this.myActivationType = (int)activationtype;
            this.iamImageRecognizer = iamImageRecognizer;
            this.myImageToRecognize = imageToRecognize;
        }

        //[JsonConstructor]
        public NetworkJsonFormat(int layerCount, int[] neuronCount, double[][][] weights, double[][] biases, int activationType, bool iamImageRecognizer, int imageToRecognize)
        {
            this.LayerCount = layerCount;
            this.NeuronCount = neuronCount;
            this.Weights = weights;
            this.Biases = biases;
            this.myActivationType = activationType;
            this.iamImageRecognizer = iamImageRecognizer;
            this.myImageToRecognize = imageToRecognize;
        }
    }
}