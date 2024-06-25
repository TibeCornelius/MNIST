namespace Ai.MNIST.NeuralNetworks.TrainingResults
{
    public class TrainingBatch
    {
        public List<ImageData> ImageData;
        public int CorrectGuesses;
        public double TotalAverageCost;
        public int TrainingSession;
        public TrainingBatch( int TrainingSession )
        {
            this.TrainingSession = TrainingSession;
            this.ImageData = new List<ImageData>();
            this.CorrectGuesses = new int();
            this.TotalAverageCost = new double();
        }
    }
}