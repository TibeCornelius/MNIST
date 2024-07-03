using System.Text.Json;

namespace Ai.MNIST.NeuralNetworks
{
    public enum ActivationFunctionOptions
    {
        Sigmoid,
        ReLU,
    }

    public sealed class Layer
    {
        public double[,] WeightsPreviousLayer;
        public double[,] WeightsGradient;
        public double[,] Velocity;//Used for Momentum base gradient descent

        public StNeuron[] StNeurons;
        private Network ParentNetwork;
        private int NeuronAmmount;
        private int LayerLevel;
        public delegate double ActivationFunction( double intput );
        private readonly ActivationFunction myActivationFunction;
        private readonly ActivationFunction myActivationFunctionDx;
 
        public Layer( Network _ParentNetwork, int _NeuronAmmount, int _LayerLevel, ActivationFunctionOptions activationFunction )
        {
            switch( activationFunction )
            {
                case ActivationFunctionOptions.Sigmoid:
                    this.myActivationFunction = ActivationFunctions.Sigmoid;
                    this.myActivationFunctionDx = ActivationFunctions.SigmoidDx;
                    break;
                case ActivationFunctionOptions.ReLU:
                    this.myActivationFunction = ActivationFunctions.ReLU;
                    this.myActivationFunctionDx = ActivationFunctions.ReLUDx;
                    break;
                default : 
                    throw new ArgumentOutOfRangeException();
            }
            this.StNeurons = initialize_Neurons( _NeuronAmmount, activationFunction );
            this.WeightsPreviousLayer = initialize_Weights( _ParentNetwork, LayerLevel, _NeuronAmmount, activationFunction ); 
            this.Velocity = initializeVelocity( _NeuronAmmount );          
            
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

        

        public Layer( Network _ParentNetwork, double[,] Wheigts, double[] Biases, int _LayerLevel, int _NeuronAmmount, bool iamFinalLayer, ActivationFunctionOptions activationFunction )
        {//Gets executed when loading network from json file
            this.ParentNetwork = _ParentNetwork;
            this.LayerLevel = _LayerLevel;
            this.StNeurons = initialize_NeuronsJson( Biases );
            this.WeightsPreviousLayer = Wheigts;
            this.Velocity = initializeVelocity( _NeuronAmmount );
            switch( activationFunction )
            {
                case ActivationFunctionOptions.Sigmoid:
                    this.myActivationFunction = ActivationFunctions.Sigmoid;
                    this.myActivationFunctionDx = ActivationFunctions.SigmoidDx;
                    break;
                case ActivationFunctionOptions.ReLU:
                    this.myActivationFunction = ActivationFunctions.ReLU;
                    this.myActivationFunctionDx = ActivationFunctions.ReLUDx;
                    break;
                default : 
                    throw new ArgumentOutOfRangeException();
            }
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
        private double[,] initializeVelocity( int neuronAmmount )
        {
            int inputSize = LayerLevel != 0 ? ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount : 28 * 28;
            double[,] Velocity = new double[ inputSize, neuronAmmount ];
            return Velocity;
        }
        private StNeuron[] initialize_Neurons( int NeuronAmmount, ActivationFunctionOptions activationType )
        {
            StNeuron[] NeuronArray = new StNeuron[NeuronAmmount];
            Random random = new Random();
            for( int index = 0 ; index < NeuronAmmount ; index++ )
            {
                double Rbiases = 0;
                switch( activationType )
                {
                    case ActivationFunctionOptions.Sigmoid:
                        Rbiases = random.NextDouble() * 2 - 1 ;
                        break;
                    case ActivationFunctionOptions.ReLU:
                        Rbiases = 0.01;;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
                NeuronArray[ index ] = new StNeuron{
                    biases = Rbiases,
                };

            }
            return NeuronArray;
        }

        private double[,] initialize_Weights( Network ParentNetwork, int LayerLevel, int NeuronAmmount, ActivationFunctionOptions activationType )
        {
            int inputSize = LayerLevel != 0 ? ParentNetwork.NetworkLayers[ LayerLevel - 1 ].NeuronAmmount : 28 * 28;
            double stddev = Math.Sqrt(2.0 / inputSize);
            double[,] weights = new double[ inputSize, NeuronAmmount ];
            Random RandomNumber = new Random();

            switch( activationType )
            {
                case ActivationFunctionOptions.ReLU:
                    for ( int indexPreviousLayer = 0; indexPreviousLayer < inputSize; indexPreviousLayer++ )
                    {
                        for (int indexThisLayer = 0; indexThisLayer < NeuronAmmount; indexThisLayer++ )
                        {
                            weights[ indexPreviousLayer, indexThisLayer ] = RandomNumber.NextGaussian( 0, stddev );
                        }
                    }
                    return weights;
                case ActivationFunctionOptions.Sigmoid:
                    for( int indexPreviousLayer = 0 ; indexPreviousLayer < inputSize ; indexPreviousLayer++ )
                    {
                        for( int indexThisLayer = 0 ; indexThisLayer < NeuronAmmount ; indexThisLayer++ )
                        {
                            weights[ indexPreviousLayer, indexThisLayer ] = RandomNumber.NextDouble() - 0.5;
                        }
                    }
                    return weights;
                default:
                    throw new ArgumentOutOfRangeException();
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
                StNeurons[ indexNeuron ].output = myActivationFunction( StNeurons[ indexNeuron ].input );
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
                double activationDerrivative = myActivationFunctionDx( StNeurons[ index ].input );
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
                newNodeValue *= myActivationFunctionDx( StNeurons[ newNodeindex ].input );
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
        
        public void ApplyGradients( double LearningRate, double HyperParameterTuner )
        {
            int NeuronGoingIn = 784;
            if ( LayerLevel - 1 >= 0 )
            {
                NeuronGoingIn = ParentNetwork.LiNetwork[ LayerLevel - 1 ];
            }

            Parallel.For( 0, NeuronAmmount, nodeOut =>
            {
                StNeurons[ nodeOut ].biases -= StNeurons[ nodeOut ].biasesGradient * LearningRate;
                for( int nodeIn = 0 ; nodeIn < NeuronGoingIn ; nodeIn++ )
                {

                    Velocity[ nodeIn, nodeOut ] = HyperParameterTuner * Velocity[ nodeIn, nodeOut ] + ( WeightsGradient[ nodeIn, nodeOut ] * LearningRate );
                    
                    WeightsPreviousLayer[ nodeIn, nodeOut ] -= Velocity[ nodeIn, nodeOut ];
                }
            });
        }
        
        public void ResetGradients()
        {
            WeightsGradient = new double[ WeightsGradient.GetLength( 0 ), WeightsGradient.GetLength( 1 ) ];
        }

        #endregion
    }
}