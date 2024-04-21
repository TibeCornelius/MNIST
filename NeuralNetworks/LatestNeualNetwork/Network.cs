namespace NeuralNetworks
{
    namespace LatestNeuralNetwork
    {
        public class Network
        {
            private List<int> NetworkLayerWithNeurons;
            public List<Layer> NetworkLayers;
            public Network( List<int> Network )
            {
                this.NetworkLayerWithNeurons = Network;
                this.NetworkLayers = initialize_Layers( Network );
            }

            private List< Layer > initialize_Layers( List<int> Network )
            {
                List<Layer> layers = new List<Layer>();
                int index = 1 ; 
                foreach ( var neuronAmmount in Network )
                {
                    layers.Add( new Layer( this, neuronAmmount, index ) );
                    index++;
                }

                return layers;
            }
        }
    }
}