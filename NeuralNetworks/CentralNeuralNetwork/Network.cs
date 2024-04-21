using System;

namespace NeuralNetworks
{
    namespace CentralNeuralNetwork
    {//First Itteration NeuralNetwork
        public class Network
        {
            private int AmmountofLayers;
            private Layer[] NetworkLayers;
            private List<int> AmmountofNeurons;
            public Network( int AmmountofLayers, List<int> AmmountofNeurons )
            {
                this.AmmountofLayers = AmmountofLayers;
                this.NetworkLayers = new Layer[ AmmountofLayers ];
                this.AmmountofNeurons = AmmountofNeurons;
                InitializeLayers();
            }

            private void ByteInput( byte[,] image )
            {
                for( int row = 0 ; row < 28 ; row++ )
                {
                    for( int column = 0 ; column < 28 ; column++ )
                    {
                        NetworkLayers[ 0 ].ListNeuronOutputs.Add( ByteToNeuronInput(image[ row, column ]) );
                    }
                }
            }

            
            public void ImportImage( byte[,] image, string imageLabel )
            {
                int index = 0; 
                foreach ( Layer Layer in NetworkLayers )
                {
                    if ( index != 0 )
                    {
                        Layer.ListNeuronOutputs =  Layer.CalculateOutputs( Layer.previousLayer.ListNeuronOutputs ) ;
                    }
                    else
                    {
                        index++;
                        ByteInput( image );
                    }
                }
                

                Console.WriteLine( "Actual Value = " + imageLabel);

                Cost( imageLabel );
                UpdateLayersWeights();
                ResetLayerNeuronOutputs();
            }

            private double Cost( string imageLabel)
            {
                double cost = 0;
                Layer outputLayer = NetworkLayers[ NetworkLayers.Length - 1 ];
                List<double> outputs = GetNetWorkOutput();
                double[] ExcpectedOutput = GetCorrectOuput( Convert.ToInt16( imageLabel ) );

                for ( int nodeOut = 0 ; nodeOut < 10 ; nodeOut++ )
                {
                    cost += NodeCost( outputs[ nodeOut ] , ExcpectedOutput[ nodeOut ]  );
                }

                return cost;
            }

            private double NodeCost( double Output, double ExpectedOutput )
            {
                double error = Output - ExpectedOutput;

                return error * error;
                //double[] NodeCost = new double[Output.Count];
                //for( int index = 0 ; index < 10 ; index++ )
                //{
                //    double error = Output[ index ] - ExpectedOutput[ index ];
                //    NodeCost[ index ] = error * error;
                //}
                //return NodeCost;
            }
            private void UpdateLayersWeights()
            {

            }

            private void ResetLayerNeuronOutputs()
            {
                foreach ( Layer Layer in NetworkLayers )
                {
                    Layer.ListNeuronOutputs.Clear();
                }
            }
            private double[] GetCorrectOuput( int imageLabel )
            {
                double[] CorrectOutput = new double[10];
                
                for (int index = 0 ; index < 10 ; index++ )
                {
                    if ( index != imageLabel )
                    {
                        CorrectOutput[ index ] = 0 ;
                    }
                    else
                    {
                        CorrectOutput[ index ] = 1 ; 
                    }
                }
                return CorrectOutput;
            }
            private List<double> GetNetWorkOutput()
            {
                double highestneuronoutput = 0;
                int highest_neuron = 0;
                List<double> outputNeurons = new List<double>();
                for( int index = 0 ; index < 10 ; index++ )
                {
                    
                }
                foreach ( Layer Layer in NetworkLayers )
                {
                    if( Layer.nextLayer == null )
                    {
                        outputNeurons = Layer.ListNeuronOutputs;
                        
                        for( int index = 0 ; index < 10 ; index++ )
                        {

                            if( Layer.ListNeuronOutputs[ index ] > highestneuronoutput )
                            {
                                highestneuronoutput = Layer.ListNeuronOutputs[ index ];
                                highest_neuron = index;
                            }
                        }
                    }
                }
                
                Console.WriteLine("Network Output = " +  highest_neuron );
                return outputNeurons;
            }


            private double ByteToNeuronInput( double WeightedInput )
            {
                double returnvalue = WeightedInput / 255;
                return returnvalue;
                //1 / ( 1 + Math.Exp( -WeightedInput));
            }

            private void InitializeLayers()
            {
                for( int index = 0 ; index < AmmountofLayers ; index++ )
                {
                    if( index > 0 )
                    {
                        NetworkLayers[ index ] = new Layer( AmmountofNeurons[ index ], NetworkLayers[ index - 1 ] );
                        NetworkLayers[ index - 1 ].nextLayer = NetworkLayers[ index ];
                    }
                    else if ( index == 0 )
                    {
                        NetworkLayers[ index ] = new Layer( AmmountofNeurons[ index ] );
                    }
                }
            }
        }
    }
}