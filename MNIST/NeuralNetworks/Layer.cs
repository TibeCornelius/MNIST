using System.Text.Json;

namespace Ai.MNIST.NeuralNetworks
{

    public sealed class Layer
    {
        public double[,] WeightsPreviousLayer;
        public double[,] WeightsGradient;

        public StNeuron[] StNeurons;
        private Network ParentNetwork;
        int NeuronAmmount;
        private int LayerLevel;
 
        public Layer( Network _ParentNetwork, int _NeuronAmmount, int _LayerLevel )
        {

            this.StNeurons = initialize_Neurons( _NeuronAmmount );
            this.WeightsPreviousLayer = initialize_Weights( _ParentNetwork, LayerLevel, _NeuronAmmount );            
            
            if( _LayerLevel == 0 )
            {
                this.WeightsGradient = new double[ 784, _NeuronAmmount ];
            }
            else
            {
                this.WeightsGradient = new double[ _ParentNetwork.LiNetwork[ _LayerLevel - 1 ], _NeuronAmmount ];
            }
            this.NeuronAmmount = _NeuronAmmount;
            this.ParentNetwork = _ParentNetwork;
            this.LayerLevel = _LayerLevel;
        }

        public Layer( Network _ParentNetwork, double[,] Wheigts, double[] Biases, int _LayerLevel, int _NeuronAmmount )
        {//Gets executed when loading network from json file
            this.ParentNetwork = _ParentNetwork;
            this.LayerLevel = _LayerLevel;
            this.StNeurons = initialize_NeuronsJson( Biases );
            this.WeightsPreviousLayer = Wheigts;
            if( _LayerLevel == 0 )
            {
                this.WeightsGradient = new double[ 784, _NeuronAmmount ];
            }
            else
            {
                this.WeightsGradient = new double[ _ParentNetwork.LiNetwork[ _LayerLevel - 1 ], _NeuronAmmount ];
            }
        }

        




        #region Json
        public double[] GetBiasesToArray()
        {
            double[] biases = new double[ StNeurons.Length ];
            int index = 0;
            foreach( StNeuron neuron in StNeurons )
            {
                biases[ index ] = neuron.biases;
                index++;
            }
            return biases;
        }
        public void CreateJson( string outputlocation )
        {
            CreateWheigtsJson( outputlocation );
            CreateBiasesJson( outputlocation );
            CreateNeuronCountJson( outputlocation );
        }

        private void CreateNeuronCountJson(string outputlocation)
        {
            string NeuronCountJson = JsonSerializer.Serialize( NeuronAmmount );
            string WheightsDestination = ".\\SavedSettings\\" + outputlocation + "\\Layer" + ( LayerLevel + 1 ) + "NeuronCount.json";
            File.WriteAllText( WheightsDestination, NeuronCountJson );
        }

        private void CreateWheigtsJson( string outputlocation )
        {
            double[][] JaggedWheigtArray = ConvertToJaggedArray( WeightsPreviousLayer );
            string JsonWheights = JsonSerializer.Serialize( JaggedWheigtArray );
            string WheightsDestination = ".\\SavedSettings\\" + outputlocation + "\\Layer" + ( LayerLevel + 1 ) + "Wheights.json";
            File.WriteAllText( WheightsDestination, JsonWheights );
        }

        private void CreateBiasesJson( string outputlocation )
        {
            double[] Biases = new double[ NeuronAmmount ];
            int index = 0;
            foreach ( StNeuron neuron in StNeurons )
            {
                Biases[ index ] = neuron.biases;
                index++;
            }
            string JsonBiases = JsonSerializer.Serialize( Biases );
            string BiasesDestination = ".\\SavedSettings\\" + outputlocation + "\\Layer" + ( LayerLevel + 1 ) + "Biases.json";
            File.WriteAllText( BiasesDestination, JsonBiases );
        }

        private double[][] ConvertToJaggedArray( double[,] Array2D )
        {
            int rows = Array2D.GetLength(0);
            int cols = Array2D.GetLength(1);
            double[][] jaggedArray = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                jaggedArray[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    jaggedArray[i][j] = Array2D[i, j];
                }
            }

            return jaggedArray;
        }
        #endregion
        #region Initialization

        private StNeuron[] initialize_NeuronsJson( double[] biases )
        {
            int NeuronCount = biases.Length;
            StNeuron[] NeuronArray = new StNeuron[ NeuronCount ];
            for( int neuronIndex = 0 ; neuronIndex < NeuronCount ; neuronIndex++ )
            {
                NeuronArray[ neuronIndex ] = new StNeuron
                {
                    biases = biases[ neuronIndex ],
                };
            }

            return NeuronArray;
        }
        
        private double[,] ReConvertJaggedArray( double[][] aaWheights )
        {
            int rowCount = aaWheights.Length;
            int colCount = aaWheights.Max(innerArray => innerArray.Length);

            double[,] Wheigts = new double[ rowCount, colCount ];

            for( int row = 0 ; row < rowCount ; row++ )
            {
                for( int column = 0 ; column < colCount ; column++ )
                {
                    Wheigts[ row, column ] = aaWheights[ row ][ column ];
                }
            }

            return Wheigts;
        }
        private StNeuron[] initialize_Neurons( int NeuronAmmount )
        {
            StNeuron[] NeuronArray = new StNeuron[NeuronAmmount];
            Random random = new Random();
            for( int index = 0 ; index < NeuronAmmount ; index++ )
            {
                double Rbiases = random.NextDouble() * 2 - 1 ;
                NeuronArray[ index ] = new StNeuron{
                    biases = Rbiases,
                };

            }
            return NeuronArray;
        }

        private double[,] initialize_Weights( Network ParentNetwork, int LayerLevel, int NeuronAmmount )
        {
            if( LayerLevel != 0 )
            {
                double[,] weights = new double[ ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount, NeuronAmmount ];
                Random RandomNumber = new Random();
                for( int indexPreviousLayer = 0 ; indexPreviousLayer < ParentNetwork.LiNetwork[ LayerLevel - 1 ] ; indexPreviousLayer++ )
                {
                    for( int indexThisLayer = 0 ; indexThisLayer < NeuronAmmount ; indexThisLayer++ )
                    {
                        weights[ indexPreviousLayer, indexThisLayer ] = RandomNumber.NextDouble() - 0.5;
                    }
                }
                return weights; 
            }
            else
            {                    
                double[,] weights = new double[ 28 * 28, NeuronAmmount ];
                Random RandomNumber = new Random();
                for( int indexPreviousLayer = 0 ; indexPreviousLayer < 28 * 28; indexPreviousLayer++ )
                {
                    for( int indexThisLayer = 0 ; indexThisLayer < NeuronAmmount ; indexThisLayer++ )
                    {
                        weights[ indexPreviousLayer, indexThisLayer ] = RandomNumber.NextDouble() - 0.5;
                    }
                }
                return weights; 
            }
        }

        #endregion
        #region inputs
        
        public void CalculateInputsEveryNeuron( ImportedImage StImportedImage )
        {
            if( LayerLevel != 0 )
            {
                Parallel.For( 0, StNeurons.Length, indexNeuron  =>
                {
                    RefStNeuron prevLayerNeurons = ParentNetwork.GetNeuronsPrevLayer( LayerLevel );
                    double input = StNeurons[ indexNeuron ].biases;
                    for( int index = 0 ; index < ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons.Length ; index++ )
                    {
                        input += prevLayerNeurons.Neurons[ index ].output * WeightsPreviousLayer[ index, indexNeuron ];
                    }
                    StNeurons[ indexNeuron ].input = input;
                });
            }
            else
            {
                Parallel.For( 0, StNeurons.Length, indexNeuron =>
                {
                    double input = StNeurons[ indexNeuron ].biases;
            
                    for( int index = 0 ; index < 784 ; index++ )
                    {
                        input += StImportedImage.output[ index ] * WeightsPreviousLayer[ index, indexNeuron ];
                    }
                    StNeurons[ indexNeuron ].input = input;
                });
            }
        }

        public void CalculateOutputs()
        {
            for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
            {
                StNeurons[ indexNeuron ].output = ActivationFunction.Sigmoid( StNeurons[ indexNeuron ].input );
            }
        }
        
        #endregion
        #region Gradient

        public double[] NodeValuesFinalLayer( double[] excpectedOutputs )
        {//Calculates the partial derrivative of the activation in respect to the output final layer only
            double[] nodeValues = new double[ excpectedOutputs.Length ];

            for( int index = 0 ; index < nodeValues.Length ; index++ )
            {
                double CostDerrivative = NodeCostDerrivative( StNeurons[ index ].output, excpectedOutputs[ index ]);
                double activationDerrivative = ActivationFunction.SigmoidDx( StNeurons[ index ].input );
                nodeValues[ index ] = activationDerrivative * CostDerrivative;
            }
            return nodeValues;
        }
        public double[] CalculateHiddenLayersNodeValues( Layer oldLayer, double[] oldNodeValues )
        {
            double[] newNodeValues = new double[ NeuronAmmount ];

            Parallel.For( 0, NeuronAmmount, newNodeindex =>
            {
                double newNodeValue = new double();
                for( int oldNodeIndex = 0 ; oldNodeIndex < oldNodeValues.Length ; oldNodeIndex++ )
                {
                    double WeightedInput = oldLayer.WeightsPreviousLayer[ newNodeindex, oldNodeIndex ];
                    newNodeValue += WeightedInput * oldNodeValues[ oldNodeIndex ]; 
                }
                newNodeValue *= ActivationFunction.SigmoidDx( StNeurons[ newNodeindex ].input );
                newNodeValues[ newNodeindex ] = newNodeValue;
            });
            return newNodeValues;
        }

        
        private double NodeCostDerrivative(double output, double ExcpectedOutput)
        {
            return 2 * ( output - ExcpectedOutput );
        }

        public void UpdateGradientLastLayer( double[] nodeValues, ImportedImage StImportedImage )
        {
            int NodesOutAmmount = 784;

            Parallel.For( 0, nodeValues.Length, nodeOut =>
            {
                for( int nodeIn = 0 ; nodeIn < NodesOutAmmount ; nodeIn++ )
                {
                    double NodeOutput = StImportedImage.output[ nodeIn ];
                    
                    double derrivativeCostWeight = NodeOutput * nodeValues[ nodeOut ];
                    WeightsGradient[ nodeIn, nodeOut ] += derrivativeCostWeight;
                }
                double derrivativeCostBiases = 1 * nodeValues[ nodeOut ];
                StNeurons[ nodeOut ].biasesGradient = derrivativeCostBiases;
            });
        }
        public void UpdateGradient( double[] nodeValues )
        {
            int NodesOutAmmount = ParentNetwork.LiNetwork[ LayerLevel - 1 ];
            Parallel.For( 0, nodeValues.Length, nodeOut =>
            {
                RefStNeuron prevLayerNeurons = ParentNetwork.GetNeuronsPrevLayer( LayerLevel ); 
                for( int nodeIn = 0 ; nodeIn < NodesOutAmmount ; nodeIn++ )
                {
                    double NodeOutput = prevLayerNeurons.Neurons[ nodeIn ].output;
                    double derrivativeCostWeight = NodeOutput * nodeValues[ nodeOut ];
                    WeightsGradient[ nodeIn, nodeOut ] += derrivativeCostWeight;
                }
                double derrivativeCostBiases = 1 * nodeValues[ nodeOut ];
                StNeurons[ nodeOut ].biasesGradient = derrivativeCostBiases;
            });
            for( int nodeOut = 0 ; nodeOut < nodeValues.Length ; nodeOut++ )
            {
                
            }
        }
        #endregion
        #region Update Weights, biases
        
        public void ApplyGradients( double LearningRate )
        {
            int NeuronGoingIn = 784;
            if ( LayerLevel - 1 >= 0 )
            {
                NeuronGoingIn = ParentNetwork.LiNetwork[ LayerLevel - 1 ];
            }
            //try 
            //{
            //    NeuronGoingIn = ParentNetwork.LiNetwork[ LayerLevel - 1 ];
            //}
            //catch
            //{
            //    NeuronGoingIn = 784;
            //}
            Parallel.For( 0, NeuronAmmount, nodeOut =>
            {
                StNeurons[ nodeOut ].biases -= StNeurons[ nodeOut ].biasesGradient * LearningRate;
                for( int nodeIn = 0 ; nodeIn < NeuronGoingIn ; nodeIn++ )
                {
                    WeightsPreviousLayer[ nodeIn, nodeOut ] -= WeightsGradient[ nodeIn, nodeOut ] * LearningRate;
                }
            });
        }
        
        public void ResetGradients()
        {
            Parallel.For( 0, WeightsGradient.GetLength( 1 ), nodeOut =>
            {
                for( int nodeIn = 0 ; nodeIn < WeightsGradient.GetLength( 0 ) ; nodeIn++ )
                {
                    WeightsGradient[ nodeIn, nodeOut ] = 0;
                }
                StNeurons[ nodeOut ].biasesGradient = 0;
            });
        }

        #endregion
    }
}