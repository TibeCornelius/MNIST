namespace NeuralNetworks
{
    namespace LatestNeuralNetwork
    {
        public class Layer
        {
            double[,] WeightsPreviousLayer;
            StNeuron[] StNeurons;
            private Network ParentNetwork;
            int NeuronAmmount;
            private int LayerLevel;
            public Layer( Network ParentNetwork, int NeuronAmmount, int LayerLevel )
            {
                this.StNeurons = initialize_Neurons( NeuronAmmount );
                this.NeuronAmmount = NeuronAmmount;
                this.ParentNetwork = ParentNetwork;
                this.LayerLevel = LayerLevel;
                if( LayerLevel != 0 )
                {
                    this.WeightsPreviousLayer = initialize_Weights( ParentNetwork, LayerLevel );
                }
                else
                {
                    this.WeightsPreviousLayer = new double[0,0];
                }
            }

            private StNeuron[] initialize_Neurons( int NeuronAmmount )
            {
                StNeuron[] NeuronArray = new StNeuron[NeuronAmmount];
                for( int index = 0 ; index < NeuronAmmount ; index++ )
                {
                    NeuronArray[ index ] = new StNeuron();
                }
                return NeuronArray;
            }

            private double[,] initialize_Weights( Network ParentNetwork, int LayerLevel )
            {
                if( ParentNetwork.NetworkLayers[ LayerLevel - 1 ] != null )
                {
                    double[,] weights = new double[ ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount, NeuronAmmount ];
                    Random RandomNumber = new Random();
                    for( int indexPreviousLayer = 0 ; indexPreviousLayer < ParentNetwork.NetworkLayers.Count ; indexPreviousLayer++ )
                    {
                        for( int indexThisLayer = 0 ; indexThisLayer < NeuronAmmount ; indexThisLayer++ )
                        {
                            weights[ indexPreviousLayer, indexThisLayer ] = RandomNumber.NextDouble() * 2 - 1;
                        }
                    }
                    return weights; 
                }
                else
                {                    
                    return new double[0,0];
                }
            }

            private void CalculateInput( byte[,] image )
            {//Gets executed on initial Layer
                for( int row = 0 ; row < 28 ; row++ )
                {
                    for( int column = 0 ; column < 28 ; column++ )
                    {
                        StNeurons[row * 28 + column ].input =  ByteToNeuronInput(image[ row, column ]);
                    }
                }
            }
            private void CalculateInput()
            {
                double[] outputPreviousLayer = new double[ ParentNetwork.NetworkLayers[ LayerLevel - 2 ].NeuronAmmount ];
                for( int PreviousLayerNeuron = 0 ; PreviousLayerNeuron < ParentNetwork.NetworkLayers[ LayerLevel - 2 ].NeuronAmmount ; PreviousLayerNeuron++ )
                {

                }
                for( int indexNeuron = 0 ; indexNeuron < NeuronAmmount ; indexNeuron++ )
                {
                    StNeurons[ indexNeuron ].input = InputIndividualNeuron( indexNeuron );
                }
            }

            

            private double InputIndividualNeuron( int indexNeuron )
            {
                double input = 0;
                int ParentNeuronindex = 0;
                foreach( StNeuron neuron in ParentNetwork.NetworkLayers[ LayerLevel - 2 ].StNeurons )
                {
                    input += neuron.input * WeightsPreviousLayer[ ParentNeuronindex, indexNeuron ];
                }
                input = 1 / ( 1 + Math.Exp( -input));
                return input;
            }

            private double ByteToNeuronInput( double WeightedInput )
            {
                double returnvalue = WeightedInput / 255;
                return returnvalue;
            }

            private void CalculateOutput()
            {

            }
        }
    }
}