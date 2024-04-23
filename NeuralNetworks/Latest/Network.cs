namespace NeuralNetworks
{
    namespace Latest
    {
        public class Network
        {
            public ImportedImage StImportedImage;
            public List<Layer> NetworkLayers;
            public List<int> LiNetwork;
            public Network( List<int> Network )
            {
                this.LiNetwork = Network;
                this.StImportedImage = new ImportedImage();
                this.NetworkLayers = new List<Layer>();
                int index = 0 ; 

                foreach( int layer in Network )
                {
                    NetworkLayers.Add( initialize_Layers( Network, index ) );
                    index++;
                }
            }

            private Layer initialize_Layers( List<int> Network, int index )
            {
                Layer layer = new Layer( this, Network[ index ], index );
                return layer;
            }

            public double[] ByteInput( byte[,] image )
            {
                double[] output = new double[ 28 * 28 ];
                for( int row = 0, total = 0 ; row < 28 ; row++ )
                {
                    for( int column = 0 ; column < 28 ; column++, total++ )
                    {
                        output[ row * 28 + column ] = ByteToNeuronInput( image[ row, column ] );
                    }
                }
                return output;
            }
            private double ByteToNeuronInput( double WeightedInput )
            {
                double returnvalue = WeightedInput / 255;
                return returnvalue;
            }
            public void Train( List<byte[,]> images, List<string> labels )
            {
                for( int imageIndex = 0 ; imageIndex < images.Count ; imageIndex++ )
                {
                    StImportedImage.image = images[ imageIndex ];
                    StImportedImage.input = images[ imageIndex ];
                    StImportedImage.output = ByteInput( images[ imageIndex ] );
                    StImportedImage.excpectedOutput = CalculateCorrectOutputs( labels[ imageIndex ] );
                    StImportedImage.label = labels[ imageIndex ];
                    foreach( Layer layer in NetworkLayers )
                    {
                        layer.CalculateInputsEveryNeuron();
                        layer.CalculateOutputs();
                    }
                    StImportedImage.cost = Cost();
                    GetHighestOutput();
                    CalculateGradients();
                    NetworkLayers[ NetworkLayers.Count - 1 ].GradientWeights();
                }
            }

            private void CalculateGradients()
            {
                for( int indexLayer = NetworkLayers.Count - 1 ; indexLayer >= 0 ; indexLayer-- )
                {
                    NetworkLayers[ indexLayer ].GradientWeights();
                }
            }
            private double GetHighestOutput()
            {
                double output = new double();
                int iOutput = 0;
                int index = 0;
                foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
                {
                    if( neuron.output > output )
                    {
                        output = neuron.output;
                        iOutput = index;
                    }
                    index++;
                }
                Console.WriteLine( "The output is " + iOutput );
                Console.WriteLine("The actual output is " + StImportedImage.label );
                return output;
            }

            public double[] CalculateCorrectOutputs( string label )
            {
                double[] output = new double[10];
                for( int index = 0 ; index < 10 ; index++ )
                {
                    output[ index ] = ( index == Convert.ToInt16( label ) ) ? 1 : 0 ;
                }

                return output;
            }
            public double Nodecost( double outPutActivation, double ExpectedOutput )
            {
                double error = outPutActivation - ExpectedOutput;
                return error * error ;
            }
            
            public double NodeCostDerrivative( double outPutActivation, double ExpectedOutput )
            {
                return 2 * ( outPutActivation - ExpectedOutput );
            }
            private double Cost( )
            {
                double[] outputs = new double[ 10 ];
                int index = 0;
                Layer outputLayer = NetworkLayers[ NetworkLayers.Count - 1 ];
                foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
                {
                    outputs[ index ] = outputLayer.StNeurons[0].input; 
                }
                double[] ExcpectedOutput = CalculateCorrectOutputs( StImportedImage.label );
                double cost = 0;

                for( int nodeOut = 0 ; nodeOut < outputs.Length ; nodeOut++ )
                {
                    cost += Nodecost( outputs[ nodeOut ], ExcpectedOutput[ nodeOut ] );
                }
                cost = cost / outputs.Length;

                return cost;
            }
        }
    }
}