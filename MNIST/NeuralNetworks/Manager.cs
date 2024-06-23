using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text.Json;


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
    public class ImportImages
    {
        public int Ammount;
        public int Itterations;
    }
    public class Manager
    {
        public Network? network;
        private List< byte[,] > bTrainingList;
        private List< string > sTrainingList;
        private List< byte[,] > bTestingList;
        private List< string > sTestingList;
        MNIST.Data.Image DataSet;
        public Manager()
        {
            this.DataSet = new MNIST.Data.Image();
            this.bTrainingList = new List< byte[,] >();
            this.sTrainingList = new List<string>();
            this.bTestingList = new List< byte[,] >();
            this.sTestingList = new List< string >();
        }
        public Image GetSingTrainingleImage()
        {
            Random random = new Random();
            int index = random.Next( bTrainingList.Count );
            foreach( MNIST.Data.Image image in MNIST.Data.MNIST.ReadTrainingData() )
            {
                bTrainingList.Add( image.Data );
                sTrainingList.Add( Convert.ToString( image.Label ) );
            }
            return new Image( bTrainingList[ index ], sTrainingList[ index ] );
        }
        public Image GetSingTestingingleImage()
        {
            Random random = new Random();
            foreach( MNIST.Data.Image image in MNIST.Data.MNIST.ReadTestData() )
            {
                bTestingList.Add( image.Data );
                sTestingList.Add( Convert.ToString( image.Label ) );
            }
            int index = random.Next( bTestingList.Count );
            return new Image( bTestingList[ index ], sTestingList[ index ] );
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


        public void LoadInNetworkFromJson()
        {
            Console.WriteLine("Do you want the standard input");
            Console.WriteLine("( Yes or true )");
            string? Input = Console.ReadLine();
            bool StandarInput = ( Input == "true" ) || ( Input == "Yes" );
            if( StandarInput )
            {
                Console.WriteLine("Give the relative Ouptut Folder");
                bool OutputNotRecieved = true;
                while ( OutputNotRecieved )
                {
                    string? relativeOuptut = Console.ReadLine();
                    if( relativeOuptut == null )
                    {
                        Console.WriteLine("No output received");
                        return;
                    }
                    else
                    {
                        string JsonLayerCount = "";
                        try
                        {
                            JsonLayerCount = File.ReadAllText( Util.StandardJsonOutput + "" + relativeOuptut + "\\LayerCount.json");
                        }
                        catch( FileNotFoundException )
                        {
                            Console.WriteLine("FileNotFound");
                            continue;
                        }
                        List<int> Neurons = new List<int>();
                        int Layercount = JsonSerializer.Deserialize<int>( JsonLayerCount );
                        for( int layer = 0 ; layer < Layercount ; layer++ )
                        {
                            string JsonNeuronCount = File.ReadAllText( Util.StandardJsonOutput + "" + relativeOuptut + "\\Layer" + (layer + 1) +"NeuronCount.json"); 
                            int NeuronCount =  JsonSerializer.Deserialize<int>( JsonNeuronCount );
                            Neurons.Add( NeuronCount );
                        }
                        bool isThisANewNetwork = false;
                        network = new Network( Neurons, isThisANewNetwork, relativeOuptut ); 
                        OutputNotRecieved = false;
                    }
                }
            }
            else
            {
                Console.WriteLine("Not yet implemented");
            }
        }


        public void SerializeWheightAndBiasesToJson()
        {
            if( network == null )
            {
                Console.WriteLine("Neuralnetworkdoes not yet exist create or import one first");
                return;            
            }
            Console.WriteLine("Give the outputlocation");
            string? FileName = Console.ReadLine();
            if( FileName != null )
            {
                network.CreateJson( FileName );
                //File.WriteAllText( "SavedWheights/" + FileName + ".json", JsonString );
            }
        }


        public List<TrainingDataOutput> ImportSetOfTestingImages( ImportImages trainingImages )
        {
            if( network == null )
            {
                return new List<TrainingDataOutput>();
            }
            foreach( MNIST.Data.Image image in MNIST.Data.MNIST.ReadTestData() )
            {
                bTestingList.Add( image.Data );
                sTestingList.Add( Convert.ToString( image.Label ) );
            }
            
            List<TrainingDataOutput> trainingResults = new List<TrainingDataOutput>();
            Random random = new Random();
            int AmmountImages;
            AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                List< byte[,] > listToTrain = new List< byte[,] >();
                List< string > sListToTrain = new List<string>();
                
                for( int index = 0 ; index < AmmountImages ; index++ )
                {
                    int randomnumber = random.Next( bTrainingList.Count );
                    listToTrain.Add( bTestingList[ randomnumber ] );
                    sListToTrain.Add( sTestingList[ randomnumber ] );
                }
                trainingResults.Add( network.Test( listToTrain, sListToTrain, Itteration + 1 ) );
            }

            bTestingList.Clear();
            sTestingList.Clear();
            return trainingResults;
        }

        public List<TrainingDataOutput> ImportSetOfTrainingImages( ImportImages trainingImages )
        {
            if( network == null )
            {
                Console.WriteLine("Neuralnetworkdoes not yet exist create or import one first");
                return new List<TrainingDataOutput>();
            }
            foreach( MNIST.Data.Image image in MNIST.Data.MNIST.ReadTestData() )
            {
                bTrainingList.Add( image.Data );
                sTrainingList.Add( Convert.ToString( image.Label ) );
            }

            List<TrainingDataOutput> trainingResults = new List<TrainingDataOutput>();
            Random random = new Random();
            int AmmountImages;
            AmmountImages = trainingImages.Ammount;
            int Itterations = trainingImages.Itterations;
            for( int Itteration = 0 ; Itteration < Itterations ; Itteration++ )
            {
                List< byte[,] > listToTrain = new List< byte[,] >();
                List< string > sListToTrain = new List<string>();
                
                for( int index = 0 ; index < AmmountImages ; index++ )
                {
                    int randomnumber = random.Next( bTrainingList.Count );
                    listToTrain.Add( bTrainingList[ randomnumber ] );
                    sListToTrain.Add( sTrainingList[ randomnumber ] );
                }
                trainingResults.Add( network.Train( listToTrain, sListToTrain, Itteration + 1 ) );
                
            }
            bTrainingList.Clear();
            sTrainingList.Clear();
            return trainingResults;
        }

    }
}


