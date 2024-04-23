using Tensorflow.Operations.Activation;

namespace NeuralNetworks
{
    namespace Latest
    {
        public class Layer
        {
            public double[,] WeightsPreviousLayer;
            public double[,]? WeightsGradient;
            public StNeuron[] StNeurons;
            private Network ParentNetwork;
            int NeuronAmmount;
            private int LayerLevel;
            public Layer( Network _ParentNetwork, int _NeuronAmmount, int _LayerLevel )
            {
                this.StNeurons = initialize_Neurons( _NeuronAmmount );
                this.NeuronAmmount = _NeuronAmmount;
                this.ParentNetwork = _ParentNetwork;
                this.LayerLevel = _LayerLevel;
                this.WeightsPreviousLayer = initialize_Weights( _ParentNetwork, LayerLevel );            
            }

            #region Initialization

            private StNeuron[] initialize_Neurons( int NeuronAmmount )
            {
                StNeuron[] NeuronArray = new StNeuron[NeuronAmmount];
                Random random = new Random();
                for( int index = 0 ; index < NeuronAmmount ; index++ )
                {
                    NeuronArray[ index ] = new StNeuron{
                        biases = random.NextDouble() * 2 - 1,
                    };

                }
                return NeuronArray;
            }

            private double[,] initialize_Weights( Network ParentNetwork, int LayerLevel )
            {
                if( LayerLevel != 0 )
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
                    double[,] weights = new double[ 28 * 28, NeuronAmmount ];
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
            }

            #endregion
            #region inputs
            
            public void CalculateInputsEveryNeuron()
            {
                if( LayerLevel != 0 )
                {
                   for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
                   {
                        StNeurons[ indexNeuron ].input = InputSingleNeuron( indexNeuron, StNeurons[ indexNeuron ] );
                   }
                }
                else
                {
                   for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
                   {
                        StNeurons[ indexNeuron ].input = FirstInputSingleNeuron( indexNeuron, StNeurons[ indexNeuron ] );
                   }
                }
            }

            private double InputSingleNeuron(int indexNeuron, StNeuron stNeuron)
            {
                double input = StNeurons[ indexNeuron ].biases;

                for( int index = 0 ; index < ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons.Length ; index++ )
                {
                    input += ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons[ index ].output * WeightsPreviousLayer[ index, indexNeuron ];
                }
                return input;
            }

            private double FirstInputSingleNeuron(int indexNeuron, StNeuron stNeuron)
            {
                double input = StNeurons[ indexNeuron ].biases;
                
                for( int index = 0 ; index < 784 ; index++ )
                {
                    input += ParentNetwork.StImportedImage.output[ index ] * WeightsPreviousLayer[ index, indexNeuron ];
                }
                return input;
            }

            public void CalculateOutputs()
            {
                for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
                {
                    StNeurons[ indexNeuron ].output = SigmoidFunction( StNeurons[ indexNeuron ].input );
                }
            }
            private double SigmoidFunction( double WeightedInput )
            {
                double output = 1 / ( 1 + Math.Exp( -WeightedInput));

                return output;
            }

            private double SigmoidDerrivative( double WeightedInput )
            {
                double Activation = SigmoidFunction( WeightedInput );
                return Activation * ( 1 - Activation );
            }
            #endregion
            #region Gradient
            public void GradientWeights()
            {
                WeightsGradient = new double[ ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount, NeuronAmmount];
                for( int NeuronOut = 0 ; NeuronOut < NeuronAmmount ; NeuronOut++ )
                {
                    for( int NeuronIn = 0 ; NeuronIn < ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount ; NeuronIn++ )
                    {
                        WeightsGradient[ NeuronIn, NeuronOut ] = CalculateGradient(NeuronIn, NeuronOut );
                    }
                }
            }

            private double CalculateGradient( int NeuronIn, int NeuronOut )
            {
                return 
                ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons[ NeuronIn ].output * 
                SigmoidFunction ( StNeurons[ NeuronOut ].input ) * ( 1 - SigmoidFunction ( StNeurons[ NeuronOut ].input ) ) * 
                2 * ( StNeurons[ NeuronOut ].output - ParentNetwork.StImportedImage.cost );
            }
            public double GradientNeuron()
            {
                return 0;
            }

            #endregion
        }
    }
}