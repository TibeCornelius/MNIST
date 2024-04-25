namespace NeuralNetworks
{
    namespace CentralNeuralNetwork
    {//First Itteration NeuralNetwork
        public class Neuron
        {   
            public double Activation { get { return State() ;} private set{} }
            public double biases { get; set;}
            private Layer ParentLayer;
            public Neuron( Layer ParentLayer )
            {
                this.ParentLayer = ParentLayer;
                this.biases = RandomBiases();
            }
            private double RandomBiases()
            {
                Random random = new Random();
                double output = random.NextDouble() * 2 - 1;
                return output;
            }


            private float State()
            {
                //Sigmoid Fucntion

                return new float();
            }
        }
    }
}