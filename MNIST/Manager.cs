using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text.Json;
using Ai.MNIST.NeuralNetworks.TrainingResults;
using Ai.MNIST.Data;


namespace Ai.MNIST.NeuralNetworks
{
    public struct NetworkValues
    {
        public int LayerCount;
        public int[] NeuronCount;
        
        public NetworkValues()
        {
            this.LayerCount = 0;
            this.NeuronCount = new int[0];
        }
        public void SetDefault()
        {
            this.LayerCount = 3;
            this.NeuronCount = new int[3]{ 400, 150, 10 };
        }
        public void SetCustom( int LayerCount, int[] NeuronCount )
        {
            this.LayerCount = LayerCount;
            this.NeuronCount = NeuronCount;
        }
    }
    public readonly struct Image
    {
        public readonly byte[,] ImageData;
        public readonly string Label;
        public Image( byte[,] ImageData, string Label )
        {
            this.ImageData = ImageData;
            this.Label = Label;
        }

    }
    public class ToImportImages( int count, List<byte[,]> images, List<string> Labels )
    {
        public int Count = count;
        public List<byte[,]> Images = images;
        public List<string> Labels = Labels;
    }
    public class ImportSettings
    {
        public int Ammount;
        public int Itterations;
    }
    public class Manager
    {
        public Network? network;
        Data.Data myDataSet;
        public delegate void DisplayResults( ImageData image );
        public Manager()
        {
            this.myDataSet = initializeDataSet();            
        }

        private Data.Data initializeDataSet()
        {
            List<byte[,]> bTrainingList = new List<byte[,]>();
            List<string> sTrainingList = new List<string>();
            foreach( MNISTImage image in MNIST.Data.MNIST.ReadTrainingData() )
            {
                bTrainingList.Add( image.Data );
                sTrainingList.Add( Convert.ToString( image.Label ) );
            }
            List<byte[,]> bTestingList = new List<byte[,]>();
            List<string> sTestingList = new List<string>();
            foreach( MNISTImage image in MNIST.Data.MNIST.ReadTestData() )
            {
                bTestingList.Add( image.Data );
                sTestingList.Add( Convert.ToString( image.Label ) );
            }
            return new Data.Data( bTrainingList, sTrainingList, bTestingList, sTestingList );
        }

        public Image GetSingTrainingleImage()
        {
            return myDataSet.GetSingleTrainingImage();
        }
        public Image GetSingleTestingImage()
        {
            return myDataSet.GetSingleTestingImage();
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
            network = new Network( Neurons );
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


        public List<TrainingBatch> ImportSetOfImages( ImportSettings trainingImages, Mode mode, bool iwillDisplayResults )
        {
            if( network == null )
            {
                return new List<TrainingBatch>();
            }
 
            List<TrainingBatch> trainingResults = new List<TrainingBatch>();
            int AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                ToImportImages importSettings = myDataSet.GetSetOfImages( trainingImages.Ammount, mode );
                if ( mode == Mode.Testing )
                {
                    trainingResults.Add( network.Test( importSettings, Itteration + 1 ) );
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
            return trainingResults;
        }

    }
}


