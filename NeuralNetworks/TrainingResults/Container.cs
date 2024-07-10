
namespace MNIST.NeuralNetworks.TrainingResults
{
    public struct Container
    {
        public List<TrainingBatch> OurResults;

        public Container()
        {
            this.OurResults = new List<TrainingBatch>();
        }

        internal List<TrainingSet> GetLatestTrainingSet()
        {
            return OurResults[ OurResults.Count - 1 ].TrainingSets;
        }
    }
}