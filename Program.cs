using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Ai.MNIST.Data;
using Tensorflow;
using System.Text.Json;

namespace Ai.MNIST.NeuralNetworks
{
    class Program
    {
        Network? network;
        private List< byte[,] > bTrainingList;
        private List< string > sTrainingList; 
        private List< byte[,] > bTestingList;
        private List< string > sTestingList;
        MNIST.Data.Image DataSet;
        public Program()
        {
            this.DataSet = new MNIST.Data.Image();
            this.bTrainingList = new List< byte[,] >();
            this.sTrainingList = new List<string>();
            this.bTestingList = new List< byte[,] >();
            this.sTestingList = new List< string >();
        }

        public void Sequence()
        {
            bool Running = true;
            while( Running )
            {
                Console.WriteLine("Choose What to do");
                Console.WriteLine("1 --> Import Sef of Training Images");
                Console.WriteLine("2 --> Import Set of Testing Images");
                Console.WriteLine("3 --> Serialize Current Wheights and Biases");
                Console.WriteLine("4 --> Load a Network from a json File");
                Console.WriteLine("5 --> Start new Network");
                Console.WriteLine("6 --> Start StandartNetwork");
                Console.WriteLine("7 --> ExitProgram ");
                
                int Choice = Convert.ToInt16( Console.ReadLine() );
                switch( Choice )
                {
                    case 1:
                        ImportSetOfTrainingImages();
                        break;
                    case 2:
                        ImportSetOfTestingImages();
                        break;
                    case 3:
                        SerializeWheightAndBiasesToJson();
                        break;
                    case 4:
                        LoadInNetworkFromJson();
                        break;
                    case 5:
                        StartNewNetwork();
                        break;
                    case 6:
                        StartNewNetwork( true );
                        break;
                    case 7:
                        Running = false;
                        break;
                    default:
                        Console.WriteLine("Invalid Choice");
                        break;
                }
            }
        }

        private void StartNewNetwork( bool StandartNetwork = false)
        {
            int AmmountOfLayers = 3;
            List<int> Neurons = new List<int>();
            if( !StandartNetwork )
            {
                Console.WriteLine("Chose the Ammount of Layers");
                bool inValidAmmount = true;
                while ( inValidAmmount )
                {
                    AmmountOfLayers = Convert.ToInt16( Console.ReadLine() ) - 1;
                    if( AmmountOfLayers > 0 && AmmountOfLayers < 5 )
                    {
                        inValidAmmount = false;
                    }
                    else
                    {
                        Console.WriteLine( "EnterValidAmmount" );
                    }
                }
                Console.WriteLine("Enter Ammount of neurons in each layer");

                for( int layer = 0 ; layer < AmmountOfLayers ; layer++ )
                {
                    Console.WriteLine($"Hidden Layer { layer }");
                    int AmmountofNeurons = Convert.ToInt16( Console.ReadLine() );
                    Neurons.Add( AmmountofNeurons );
                }
            }
            else
            {
                Neurons.Add( 400 );
                Neurons.Add( 150 );            
            }
            Neurons.Add( 10 );
            network = new Network( Neurons );
        }

        private void LoadInNetworkFromJson()
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

        private void SerializeWheightAndBiasesToJson()
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


        private void ImportSetOfTestingImages()
        {
            if( network == null )
            {
                Console.WriteLine("Neuralnetworkdoes not yet exist create or import one first");
                return;
            }
            foreach( MNIST.Data.Image image in Data.MNIST.ReadTestData() )
            {
                bTestingList.Add( image.Data );
                sTestingList.Add( Convert.ToString( image.Label ) );
            }

            Random random = new Random();
            int AmmountImages;
            Console.WriteLine("How Many images do you want to import?");
            AmmountImages = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("How Many Itteration do you want to run?");
            int Itterations = Convert.ToInt16( Console.ReadLine() );
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
                network.Test( listToTrain, sListToTrain, Itteration + 1 );
            }

            bTestingList.Clear();
            sTestingList.Clear();
        }

        private void ImportSetOfTrainingImages()
        {
            if( network == null )
            {
                Console.WriteLine("Neuralnetworkdoes not yet exist create or import one first");
                return;
            }
            foreach( MNIST.Data.Image image in Data.MNIST.ReadTrainingData() )
            {
                bTrainingList.Add( image.Data );
                sTrainingList.Add( Convert.ToString( image.Label ) );
            }
            Random random = new Random();
            int AmmountImages;
            Console.WriteLine("How Many images do you want to import?");
            AmmountImages = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("How Many Itteration do you want to run?");
            int Itterations = Convert.ToInt16( Console.ReadLine() );
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
                network.Train( listToTrain, sListToTrain, Itteration + 1 );
            }
            bTrainingList.Clear();
            sTrainingList.Clear();
        }

        static void Main(string[] args)
        {
            Program Main = new Program();
            Main.Sequence();
        }
    }
}


