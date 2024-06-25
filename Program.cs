using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Ai.MNIST.Data;
using Tensorflow;
using System.Text.Json;
using Ai.MNIST.NeuralNetworks;
using Ai.MNIST.NeuralNetworks.TrainingResults;

namespace Ai.MNIST.Terminal
{
    class Program
    {
        private Manager myManager;

        public Program()
        {
            this.myManager = new Manager();
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
            NetworkValues settings = new();
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
                settings.SetCustom( AmmountOfLayers, Neurons.ToArray() );
            }
            else
            {
                settings.SetDefault();          
            }
            
            myManager.StartNewNetwork( settings );
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
                        string JsonString = string.Empty;
                        try
                        {
                            JsonString = File.ReadAllText( OutPuts.StandardJsonOutput + "" + relativeOuptut + "\\LayerCount.json");
                        }
                        catch( FileNotFoundException )
                        {
                            Console.WriteLine("FileNotFound");
                            goto LoopEnd;
                        }
                        try
                        {
                            NetworkJsonFormat JsonSettings = JsonSerializer.Deserialize<NetworkJsonFormat>( JsonString );
                            myManager.LoadInNetworkFromJson( JsonSettings ); 
                        }
                        catch( Exception )
                        {
                            goto LoopEnd;
                        }
                        LoopEnd:;
                    }
                }
            }
            else
            {
                Console.WriteLine("Give the absolut location of the .json file");
                bool OutputNotRecieved = true;
                while ( OutputNotRecieved )
                {
                    string? Location = Console.ReadLine();
                    if( Location == null )
                    {
                        Console.WriteLine("No output received");
                        return;
                    }
                    else
                    {
                        try
                        {
                            string JsonString = File.ReadAllText( Location );
                            NetworkJsonFormat? settings = JsonSerializer.Deserialize<NetworkJsonFormat>( JsonString );
                            if( settings is null )
                            {
                                throw new Exception();
                            }
                            myManager.LoadInNetworkFromJson( settings );
                        }
                        catch
                        {
                            Console.WriteLine("Something went wrong try again");
                            goto LoopEnd;
                        }
                        OutputNotRecieved = false;
                    }
                    LoopEnd:;
                }
            }
        }

        private void SerializeWheightAndBiasesToJson()
        {

            Console.WriteLine("Give the outputlocation");
            string? FileName = Console.ReadLine();
            if( FileName is not null )
            {
                myManager.SerializeWheightAndBiasesToJson( FileName );
            }
        }


        private void ImportSetOfTestingImages()
        {
            ImportImages settings = new();



            Console.WriteLine("How Many images do you want to import?");
            settings.Ammount = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("How Many Itteration do you want to run?");
            settings.Itterations = Convert.ToInt16( Console.ReadLine() );

            myManager.ImportSetOfTestingImages( settings );
        }

        private void ImportSetOfTrainingImages()
        {
            ImportImages settings = new();



            Console.WriteLine("How Many images do you want to import?");
            settings.Ammount = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("How Many Itteration do you want to run?");
            settings.Itterations = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("Do you want to display the results");
            Console.WriteLine("( Yes or true )");
            string? Input = Console.ReadLine();
            bool DisplayResults = ( Input == "true" ) || ( Input == "Yes" );

            if( DisplayResults )
            {
                myManager.network.displayBatchResults = DisplayBatchResults;
                myManager.network.displayResults = DisplayImageResults;
                myManager.ImportSetOfTrainingImages( settings, true );
            }
            else
            {
                myManager.ImportSetOfTrainingImages( settings );
            }
        }
        public void DisplayImageResults( ImageData image )
        {
            //Console.WriteLine("Hello world");
        }
        public void DisplayBatchResults( TrainingBatch trainingBatch )
        {
            Console.WriteLine( "Average Costs" + trainingBatch.TotalAverageCost );
            Console.WriteLine( "Ammount of correct guesses " + trainingBatch.CorrectGuesses );
            Console.WriteLine( "Training session " + trainingBatch.TrainingSession );
        }

        static void Main(string[] args)
        {
            Program Main = new Program();
            Main.Sequence();
        }
    }
}


