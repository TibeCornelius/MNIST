namespace NeuralNetworks
{
    namespace Latest
    {
        public struct ImportedImage
        {
            public byte[,] image;
            public string label;
            public byte[,] input;
            public double[] output;
            public double cost;
            public double[] excpectedOutput;
        }
    }
}