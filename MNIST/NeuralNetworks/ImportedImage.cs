namespace Ai.MNIST.NeuralNetworks
{
    public struct ImportedImage
    {
        public double[] nodeValuesFinalLayer;
        public byte[,] image;
        public string label;
        public byte[,] input;
        public double[] output;
        public double cost;
        public double[] excpectedOutput;
    }

}