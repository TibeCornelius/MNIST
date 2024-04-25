using System;
using Tensorflow;

namespace NeuralNetworks
{
    namespace CentralNeuralNetwork
    {//First Itteration NeuralNetwork
        public class Layer
        {



            //StNeuron[] StNeuronArray;
            public int AmmountofNeurons { get; private set; }
            public double[,]? Wheights { get; set;}
            public List<double> ListNeuronOutputs { get; set ; }
            public Layer? previousLayer;
            public Layer? nextLayer { get; set; }
            public Layer( int AmmountofNeurons, Layer previousLayer )
            {//gets initialized every layer except for the first
                this.AmmountofNeurons = AmmountofNeurons;
                this.previousLayer = previousLayer;
                //this.StNeuronArray = StNeuronArrayCreation();

                this.Wheights = initialize_RandomWeights();
                this.ListNeuronOutputs = new List<double>();
            }
            public Layer( int AmmountofNeurons )
            {//gets initialized at the fist layer no weights
                this.AmmountofNeurons = AmmountofNeurons;
                this.ListNeuronOutputs = new List<double>();
            }

            //private StNeuron[] StNeuronArrayCreation()
            //{
            //    StNeuron[] NeuronArray = new StNeuron[ AmmountofNeurons ];
            //    Random random = new Random();
            //    for ( int Neuron = 0 ; Neuron < NeuronArray.Length ; Neuron++ )
            //    {
            //        StNeuron neuron = new StNeuron();
//
            //        neuron.biases = random.NextDouble() * 2 - 1;
            //    }
//
            //    return NeuronArray;
            //}

            public List<double>? CalculateOutputs( List<double> inputs )
            {
                List<double> Activationoutputs = new List<double>();

                for( int indexNodeout = 0 ; indexNodeout < AmmountofNeurons ; indexNodeout++ )
                {
                    double WeightedInput = 0 ;  
                    //NeuronArray[ indexNodeout ].biases;
                    for( int indexNodein = 0 ; indexNodein < previousLayer.AmmountofNeurons ; indexNodein++ )
                    {
                        WeightedInput += inputs[ indexNodein ] * Wheights[ indexNodein, indexNodeout ];
                    }
                    Activationoutputs.add( ActivationFunction( WeightedInput ) ); 
                }
                return Activationoutputs;
            }
            

            private double ActivationFunction( double WeightedInput )
            {
                double output = 1 / ( 1 + Math.Exp( -WeightedInput));

                return output;
            }

            private double[,] initialize_RandomWeights()
            {

                if ( previousLayer == null )
                {
                    return new double[0,0];
                }
                else
                {
                    double[,] wheights = new double[ previousLayer.AmmountofNeurons, AmmountofNeurons ];

                    for( int indexPreviousLayer = 0 ; indexPreviousLayer < previousLayer.AmmountofNeurons ; indexPreviousLayer++ )
                    {
                        for( int indexCurrentLayer = 0 ; indexCurrentLayer < AmmountofNeurons ; indexCurrentLayer++ )
                        {
                            wheights[ indexPreviousLayer, indexCurrentLayer ] = new Random().NextDouble() * 2 - 1 ;
                        }
                    }
                    return wheights;
                } 
            }
        }
    }
}