using System.Text.Json;

namespace Ai.MNIST.NeuralNetworks
{

    public class Layer
    {
        public double[,] WeightsPreviousLayer;
        public double[,] WeightsGradient;

        public StNeuron[] StNeurons;
        private Network ParentNetwork;
        int NeuronAmmount;
        private int LayerLevel;
        public Layer( Network _ParentNetwork, int _NeuronAmmount, int _LayerLevel, bool FromJson = false, string _JsonFile = "" )
        {
            if( FromJson == false )
            {
                this.StNeurons = initialize_Neurons( _NeuronAmmount );
                this.WeightsPreviousLayer = initialize_Weights( _ParentNetwork, LayerLevel, _NeuronAmmount );            
            }
            else
            {
                this.StNeurons = initializeFromJson_Neurons( _NeuronAmmount, _JsonFile, _LayerLevel );
                this.WeightsPreviousLayer = initializeFromJson_Wheights( _JsonFile, _LayerLevel ); 
            }
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
        #region Json
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

        private StNeuron[] initializeFromJson_Neurons( int NeuronAmmount, string JsonFile, int LayerLevel  )
        {
            StNeuron[] NeuronArray = new StNeuron[ NeuronAmmount ];
            string JsonBiases = File.ReadAllText( Util.StandardJsonOutput + "" + JsonFile + "\\Layer" + ( LayerLevel + 1 ) + "Biases.json" );
            double[]? aBiases = JsonSerializer.Deserialize<double[]>( JsonBiases );
            if( aBiases == null )
            {
                throw new NullReferenceException("Biases from json where NULL");
            }
            for( int NeuronCount = 0 ; NeuronCount < NeuronAmmount ; NeuronCount++ )
            {
                StNeuron myNeuron = new StNeuron{
                    biases = aBiases[ NeuronCount ],
                };
                NeuronArray[ NeuronCount ] = myNeuron;
            }
            return NeuronArray;
        }
        private double[,] initializeFromJson_Wheights( string JsonFile, int LayerLevel )
        {
            string JsonWheights = File.ReadAllText( Util.StandardJsonOutput + "" + JsonFile + "\\Layer" + ( LayerLevel + 1 ) + "Wheights.json");
            double[][]? aaWheights = JsonSerializer.Deserialize<double[][]>( JsonWheights );
            if( aaWheights == null )
            {
                throw new NullReferenceException("Wheights from json where NULL");
            }
            double[,] Wheigts = ReConvertJaggedArray( aaWheights );
            return Wheigts;
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
                for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
                {
                    StNeurons[ indexNeuron ].input = InputSingleNeuron( indexNeuron, StNeurons[ indexNeuron ] );
                }
            }
            else
            {
                for( int indexNeuron = 0 ; indexNeuron < StNeurons.Length ; indexNeuron++ )
                {
                    StNeurons[ indexNeuron ].input = FirstInputSingleNeuron( indexNeuron, StNeurons[ indexNeuron ], StImportedImage );
                }
            }
        }

        private double InputSingleNeuron( int indexNeuron, StNeuron stNeuron )
        {
            double input = StNeurons[ indexNeuron ].biases;

            for( int index = 0 ; index < ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons.Length ; index++ )
            {
                input += ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons[ index ].output * WeightsPreviousLayer[ index, indexNeuron ];
            }
            return input;
        }

        private double FirstInputSingleNeuron( int indexNeuron, StNeuron stNeuron, ImportedImage StImportedImage )
        {
            double input = StNeurons[ indexNeuron ].biases;
            
            for( int index = 0 ; index < 784 ; index++ )
            {
                input += StImportedImage.output[ index ] * WeightsPreviousLayer[ index, indexNeuron ];
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

        public double[] NodeValuesFinalLayer( double[] excpectedOutputs )
        {//Calculates the partial derrivative of the activation in respect to the output final layer only
            double[] nodeValues = new double[ excpectedOutputs.Length ];

            for( int index = 0 ; index < nodeValues.Length ; index++ )
            {
                double CostDerrivative = NodeCostDerrivative( StNeurons[ index ].output, excpectedOutputs[ index ]);
                double activationDerrivative = SigmoidDerrivative( StNeurons[ index ].input );
                nodeValues[ index ] = activationDerrivative * CostDerrivative; 
            }
            return nodeValues;
        }
        public double[] CalculateHiddenLayersNodeValues( Layer oldLayer, double[] oldNodeValues )
        {
            double[] newNodeValues = new double[ NeuronAmmount ];

            for( int newNodeindex = 0 ; newNodeindex < newNodeValues.Length ; newNodeindex++ )
            {
                double newNodeValue = new double();
                for( int oldNodeIndex = 0 ; oldNodeIndex < oldNodeValues.Length ; oldNodeIndex++ )
                {
                    double WeightedInput = oldLayer.WeightsPreviousLayer[ newNodeindex, oldNodeIndex ];
                    newNodeValue += WeightedInput * oldNodeValues[ oldNodeIndex ]; 
                }
                newNodeValue *= SigmoidDerrivative(StNeurons[ newNodeindex ].input );
                newNodeValues[ newNodeindex ] = newNodeValue;
            }

            return newNodeValues;
        }

        
        private double NodeCostDerrivative(double output, double ExcpectedOutput)
        {
            return 2 * ( output - ExcpectedOutput );
        }

        public void UpdateGradientLastLayer( double[] nodeValues, ImportedImage StImportedImage )
        {
            int NodesOutAmmount = 784;

            for( int nodeOut = 0 ; nodeOut < nodeValues.Length ; nodeOut++ )
            {
                for( int nodeIn = 0 ; nodeIn < NodesOutAmmount ; nodeIn++ )
                {
                    double NodeOutput = StImportedImage.output[ nodeIn ];
                    
                    double derrivativeCostWeight = NodeOutput * nodeValues[ nodeOut ];
                    WeightsGradient[ nodeIn, nodeOut ] += derrivativeCostWeight;
                }
                double derrivativeCostBiases = 1 * nodeValues[ nodeOut ];
                StNeurons[ nodeOut ].biasesGradient = derrivativeCostBiases;
            }
        }
        public void UpdateGradient( double[] nodeValues )
        {
            int NodesOutAmmount = ParentNetwork.LiNetwork[ LayerLevel - 1 ];
            for( int nodeOut = 0 ; nodeOut < nodeValues.Length ; nodeOut++ )
            {
                for( int nodeIn = 0 ; nodeIn < NodesOutAmmount ; nodeIn++ )
                {
                    double NodeOutput = ParentNetwork.NetworkLayers[ LayerLevel - 1 ].StNeurons[ nodeIn ].output;
                    double derrivativeCostWeight = NodeOutput * nodeValues[ nodeOut ];
                    WeightsGradient[ nodeIn, nodeOut ] += derrivativeCostWeight;
                }
                double derrivativeCostBiases = 1 * nodeValues[ nodeOut ];
                StNeurons[ nodeOut ].biasesGradient = derrivativeCostBiases;
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
            for( int nodeOut = 0 ; nodeOut < NeuronAmmount ; nodeOut++ )
            {
                StNeurons[ nodeOut ].biases -= StNeurons[ nodeOut ].biasesGradient * LearningRate;
                for( int nodeIn = 0 ; nodeIn < NeuronGoingIn ; nodeIn++ )
                {
                    WeightsPreviousLayer[ nodeIn, nodeOut ] -= WeightsGradient[ nodeIn, nodeOut ] * LearningRate;
                } 
            }
        }
        
        public void ResetGradients()
        {
            for( int nodeOut = 0 ; nodeOut < WeightsGradient.GetLength( 1 ) ; nodeOut++ )
            {
                for( int nodeIn = 0 ; nodeIn < WeightsGradient.GetLength( 0 ) ; nodeIn++ )
                {
                    WeightsGradient[ nodeIn, nodeOut ] = 0;
                }
                StNeurons[ nodeOut ].biasesGradient = 0;
            }
        }

        #endregion
    }
}