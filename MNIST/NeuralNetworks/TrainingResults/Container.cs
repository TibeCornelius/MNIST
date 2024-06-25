namespace Ai.MNIST.NeuralNetworks.TrainingResults
{
    public struct Container
    {
        public List<TrainingBatch> OurTrainingResults;
        public List<TrainingBatch> OurTestingResults;
        public Container()
        {
            this.OurTestingResults = new List<TrainingBatch>();
            this.OurTrainingResults = new List<TrainingBatch>();
        }
    }
}