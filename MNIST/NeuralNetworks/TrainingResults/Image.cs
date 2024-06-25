namespace Ai.MNIST.NeuralNetworks.TrainingResults
{
    public class ImageData( int ImageNumber, double Cost, int Guess, byte[,] Image, StNeuron[] OutputNeurons )
    {
        public int ImageNumber = ImageNumber;
        public byte[,] Image = Image;
        public int NumberGuessed = Guess;
        public double[] Results = [];
        public StNeuron[] neuronResults = OutputNeurons;
        public bool wasGuesCorrect => ImageNumber == NumberGuessed ;
        public double Cost = Cost; 
    }
}