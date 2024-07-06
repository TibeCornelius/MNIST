namespace MNIST.NeuralNetworks.TrainingResults
{
    public struct Container
    {
        public List<TrainingBatch> OurResults;
        public List<double> OurTrainingTimes;

        public Container()
        {
            this.OurResults = new List<TrainingBatch>();
            this.OurTrainingTimes = new List<double>();
        }
    }
}