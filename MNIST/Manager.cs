using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text.Json;
using Ai.MNIST.NeuralNetworks.TrainingResults;
using Ai.MNIST.Data;
using System.Diagnostics;


namespace Ai.MNIST.NeuralNetworks
{
    public struct NetworkValues
    {
        public int LayerCount;
        public int[] NeuronCount;
        public ActivationFunctionOptions ActivationFunction;
        
        public NetworkValues()
        {
            this.LayerCount = 0;
            this.NeuronCount = new int[0];
        }
        public void SetDefault()
        {
            this.LayerCount = 3;
            this.NeuronCount = new int[3]{ 400, 150, 10 };
            this.ActivationFunction = ActivationFunctionOptions.Sigmoid;
        }
        public void SetCustom( int LayerCount, int[] NeuronCount,ActivationFunctionOptions activationFunctionOptions )
        {
            this.LayerCount = LayerCount;
            this.NeuronCount = NeuronCount;
            this.ActivationFunction = activationFunctionOptions;
        }
    }
    public readonly struct Image
    {
        public readonly byte[,] ImageData;
        public readonly int Label;
        public Image( byte[,] ImageData, int Label )
        {
            this.ImageData = ImageData;
            this.Label = Label;
        }

    }
    public readonly struct ToImportImages( int count, List<byte[,]> images, List<int> Labels )
    {
        public readonly int Count = count;
        public readonly List<byte[,]> Images = images;
        public readonly List<int> Labels = Labels;
    }
    public class ImportSettings
    {
        public int Ammount;
        public int Itterations;
    }
    public class Manager
    {
        public Network? network;
        public Network[]? NumberVerifyerNetworks;
        Data.Data myDataSet;
        public delegate void DisplayResults( ImageData image );
        public Manager()
        {
            this.myDataSet = initializeDataSet();            
        }

        private Data.Data initializeDataSet()
        {
            List<byte[,]> bTrainingList = new List<byte[,]>();
            List<int> sTrainingList = new List<int>();
            foreach( MNISTImage image in MNIST.Data.MNIST.ReadTrainingData() )
            {
                bTrainingList.Add( image.Data );
                sTrainingList.Add( Convert.ToInt16( image.Label ) );
            }
            List<byte[,]> bTestingList = new List<byte[,]>();
            List<int> sTestingList = new List<int>();
            foreach( MNISTImage image in MNIST.Data.MNIST.ReadTestData() )
            {
                bTestingList.Add( image.Data );
                sTestingList.Add( Convert.ToInt16( image.Label ) );
            }
            return new Data.Data( bTrainingList, sTrainingList, bTestingList, sTestingList );
        }

        public Image GetSingTrainingleImage( bool AddNoise )
        {
            return myDataSet.GetSingleTrainingImage( AddNoise );
        }
        public Image GetSingleTestingImage( bool AddNoise )
        {
            return myDataSet.GetSingleTestingImage( AddNoise );
        }
        public TrainingBatch ImportSingleImage( Image image )
        {
            if( network is null )
            {
                return new TrainingBatch(-1);
            }
            return network.ImportSingleImage( image );
        }
        public void StartNewNetwork( NetworkValues networkValues )
        {
            List<int> Neurons = new List<int>();
            foreach( int NeuronCount in networkValues.NeuronCount )
            {
                Neurons.Add( NeuronCount );
            }
            network = new Network( Neurons, networkValues.ActivationFunction );
        }

        public void LoadInNetworkFromJson( NetworkJsonFormat JsonSettings )
        {
            network = new Network( JsonSettings );
        }

        public void SerializeWheightAndBiasesToJson( string output, bool DefaultLocation )
        {
            if( network == null )
            {
                return;            
            }

            network.CreateJson( output, DefaultLocation ); 
        }

        public void SerializeCurrentHistory( string OutputLocation, bool DefaultLocation )
        {
            if( network is null )
            {
                return;
            }
            network.SerializeStatsToJson( OutputLocation, DefaultLocation );
        }
        public List<TrainingBatch> ImportSetOfImages( ImportSettings trainingImages, Mode mode, bool iwillDisplayResults, bool AddNoise = false )
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            if( network == null )
            {
                return new List<TrainingBatch>();
            }
 
            List<TrainingBatch> trainingResults = new List<TrainingBatch>();
            int AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                ToImportImages importSettings = myDataSet.GetSetOfImages( trainingImages.Ammount, mode, AddNoise );
                if ( mode == Mode.Testing )
                {
                    if( iwillDisplayResults )
                    {
                        trainingResults.Add( network.Test( importSettings, Itteration + 1 , iwillDisplayResults ));
                    }
                    else
                    {
                        trainingResults.Add( network.Test( importSettings, Itteration + 1 ) );
                    }
                }
                else
                {
                    if( iwillDisplayResults )
                    {
                        trainingResults.Add( network.Train( importSettings, Itteration + 1, iwillDisplayResults ) );
                    }
                    else
                    {
                        trainingResults.Add( network.Train( importSettings, Itteration + 1 ) );
                    }
                }
            }
            stopwatch.Stop();
            Console.WriteLine($"Total elapsed time --> { stopwatch.Elapsed} ");
            return trainingResults;
        }
    }
}


