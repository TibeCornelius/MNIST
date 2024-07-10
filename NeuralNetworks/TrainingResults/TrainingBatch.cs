namespace MNIST.NeuralNetworks.TrainingResults
{
    public class TrainingBatch
    {
        public List<TrainingSet> TrainingSets;
        public double TrainingTime;
        public DateTime dateTime;//Reference to when the training Batch was created
        public TrainingBatch()
        {
            this.TrainingSets = new List<TrainingSet>();
            this.dateTime = DateTime.Now;
        }

    }
}