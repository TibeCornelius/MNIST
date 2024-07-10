namespace MNIST.NeuralNetworks.TrainingResults
{
    public class TrainingSet
    {
        public List<ImageData> ImageData;
        public int CorrectGuesses;
        public double TotalAverageCost;
        public int TrainingSession;
        public double TrainingDuration;
        public double TrainingTime;
        public TrainingSet( int TrainingSession )
        {
            this.TrainingSession = TrainingSession;
            this.ImageData = new List<ImageData>();
            this.CorrectGuesses = new int();
            this.TotalAverageCost = new double();
        }
    }
}