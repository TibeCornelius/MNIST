using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text.Json;
using MNIST.NeuralNetworks.TrainingResults;
using MNIST.Data;
using System.Diagnostics;


namespace MNIST.NeuralNetworks
{
    public struct NetworkValues
    {
        public int LayerCount;
        public int[] NeuronCount;
        public ActivationFunctionOptions ActivationFunction;
        public bool isThisImageRecognizer; 
        
        public NetworkValues()
        {
            this.LayerCount = 0;
            this.NeuronCount = new int[0];
            this.isThisImageRecognizer = true;
        }
        public void SetDefault()
        {
            this.LayerCount = 3;
            this.NeuronCount = new int[3]{ 400, 150, 10 };
            this.ActivationFunction = ActivationFunctionOptions.LeakyRelu;
            this.isThisImageRecognizer = true;
        }
        public void SetCustom( int LayerCount, int[] NeuronCount, ActivationFunctionOptions activationFunctionOptions, bool isThisImageRecognizer )
        {
            this.LayerCount = LayerCount;
            this.NeuronCount = NeuronCount;
            this.ActivationFunction = activationFunctionOptions;
            this.isThisImageRecognizer = isThisImageRecognizer;
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
        public bool iamImportingToImageRecognizer;
    }
    public class Manager
    {
        public Network? network;
        public Network[]? NumberVerifyerNetworks;
        Data.Data myDataSet;
        public Container TrainingData;
        public delegate void DisplayResults( ImageData image );
        public Manager()
        {
            this.myDataSet = initializeDataSet();
            this.TrainingData = new Container();       
        }
        public ToImportImages GetSetOfImages( Mode mode, bool AddNoise, int Ammount )
        {
            ToImportImages toImportImages = myDataSet.GetSetOfImages( Ammount, mode, AddNoise ); 
            return toImportImages;
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
        public List<TrainingSet> GetLatestTrainingSet()
        {
            return TrainingData.GetLatestTrainingSet();
        }
        public bool ChangeNetworkDisplayImageResults( DisplayImageResults displayImageResults )
        {
            if( network == null )
            {
                return false;
            }
            network.displayResults = displayImageResults;
            return true;
        }
        public bool ChangeNetworkDisplaySetResults( DisplayBatchResults displayBatchResults )
        {
            if( network == null )
            {
                return false;
            }
            network.displayBatchResults = displayBatchResults;
            return true;
        }
        public Image GetSingTrainingleImage( bool AddNoise )
        {
            return myDataSet.GetSingleTrainingImage( AddNoise );
        }
        public Image GetSingleTestingImage( bool AddNoise )
        {
            return myDataSet.GetSingleTestingImage( AddNoise );
        }
        public TrainingSet ImportSingleImage( Image image )
        {
            if( network is null )
            {
                return new TrainingSet(-1);
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
            if( networkValues.isThisImageRecognizer )
            {
                network = new Network( Neurons, networkValues.ActivationFunction, true, 0, this );
            }
            else
            {
                NumberVerifyerNetworks = new Network[ 10 ];
                for( int networkIndex = 0 ; networkIndex < 10 ; networkIndex++ )
                {
                    NumberVerifyerNetworks[ networkIndex ] = new Network( Neurons, networkValues.ActivationFunction, false, networkIndex, this );
                }
            }
        }
        public void LoadInNetworkFromJson( NetworkJsonFormat JsonSettings, bool ImageVerifyer = false )
        {
            if( ImageVerifyer )
            {
                if( NumberVerifyerNetworks is null )
                {
                    NumberVerifyerNetworks = new Network[10];
                }
                NumberVerifyerNetworks[ JsonSettings.myImageToRecognize ] = new Network( JsonSettings, this );
            }
            else
            {
                network = new Network( JsonSettings, this );
            }
        }

        public bool VerifyNetworkOutputs( Image Image, int ResultToCheck )
        {
            if( NumberVerifyerNetworks == null )
            {
                throw new NullReferenceException();
            }
            TrainingSet results = NumberVerifyerNetworks[ Image.Label ].ImportSingleImage( Image );
            if( results.ImageData[ 0 ].wasGuesCorrect )
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        public void SerializeWheightAndBiasesToJson( string output, bool DefaultLocation, bool SerializeVerifyer, string FileName = "" )
        {

            if( SerializeVerifyer )
            {
                if( NumberVerifyerNetworks is null )
                {
                    return;
                }
                else
                {
                    for( int networkIndex = 0 ; networkIndex < 10 ; networkIndex++ )
                    {
                        try
                        {
                            string newoutput = output + networkIndex;
                            NumberVerifyerNetworks[ networkIndex ].CreateJson( newoutput, DefaultLocation );
                        }
                        catch
                        {

                        }
                    }
                }
            }
            else
            {
                if( network == null )
                {
                    return;            
                }
                if( FileName != "" )
                {
                    network.CreateJson( output, DefaultLocation, FileName ); 
                }
                else
                {
                    network.CreateJson( output, DefaultLocation ); 
                }
            }
        }

        public bool TrainImageVerifyer( ImportSettings trainingImages, Mode mode, bool iwillDisplayResults, bool AddNoise )
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            if( NumberVerifyerNetworks == null )
            {
                return false;
            }
 
            TrainingBatch trainingResults = new ();
            int AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                ToImportImages importSettings = myDataSet.GetSetOfImages( trainingImages.Ammount, mode, AddNoise );
                for( int NetworkIndex = 0 ; NetworkIndex < 10 ; NetworkIndex++ )
                {
                    Network networkToTrain = NumberVerifyerNetworks[ NetworkIndex ];
                    if ( mode == Mode.Testing )
                    {
                        bool iamTrainingNetwork = false;
                        trainingResults.TrainingSets.Add( networkToTrain.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                    }
                    else
                    {
                        bool iamTrainingNetwork = true;
                        trainingResults.TrainingSets.Add( networkToTrain.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                    }
                    Console.WriteLine($"Network to verify number { NetworkIndex }");
                }
            }
            stopwatch.Stop();
            Console.WriteLine($"Total elapsed time --> { stopwatch.Elapsed} ");
            return true;
        }
        public void SerializeCurrentHistory( string OutputLocation, bool DefaultLocation )
        {
            if( network is null )
            {
                return;
            }
        }
        public bool ImportSetOfImages( ImportSettings trainingImages, Mode mode, bool iwillDisplayResults, bool AddNoise = false )
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            if( network == null )
            {
                return false;
            }

            TrainingBatch trainingBatch = new();
            int AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                ToImportImages importSettings = myDataSet.GetSetOfImages( AmmountImages, mode, AddNoise );
                if ( mode == Mode.Testing )
                {
                    bool iamTrainingNetwork = false;
                    trainingBatch.TrainingSets.Add( network.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                }
                else
                {
                    bool iamTrainingNetwork = true;
                    trainingBatch.TrainingSets.Add( network.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                }
            }
            stopwatch.Stop();
            trainingBatch.TrainingTime = stopwatch.Elapsed.Milliseconds;
            TrainingData.OurResults.Add( trainingBatch );
            Console.WriteLine($"Total time elapsed { stopwatch.Elapsed }");
            return true;
        }

        public void TrainSpecificImageVerifyer( ImportSettings trainingImages, Mode mode, bool iwillDisplayResults, bool AddNoise, int ImageVerifyerToTrain )
        {
            Stopwatch stopwatch = new();
            stopwatch.Start();
            if( NumberVerifyerNetworks == null )
            {
                return;
            }
 
            TrainingBatch trainingResults = new();
            int AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                ToImportImages importSettings = myDataSet.GetSetOfImages( trainingImages.Ammount, mode, AddNoise );

                Network networkToTrain = NumberVerifyerNetworks[ ImageVerifyerToTrain ];
                if ( mode == Mode.Testing )
                {
                    bool iamTrainingNetwork = false;
                    trainingResults.TrainingSets.Add( networkToTrain.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                }
                else
                {
                    bool iamTrainingNetwork = true;
                    trainingResults.TrainingSets.Add( networkToTrain.RunImagesThroughNetwork( importSettings, iamTrainingNetwork, Itteration + 1 , iwillDisplayResults, false ) );
                }
                
            }
            stopwatch.Stop();

        }
    }
}


