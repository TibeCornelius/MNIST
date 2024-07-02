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
                Console.WriteLine("7 --> Serialize current network history");
                Console.WriteLine("8 --> Load in network history form .json");
                Console.WriteLine("9 --> ExitProgram ");
                
                int Choice = Convert.ToInt16( Console.ReadLine() );
                switch( Choice )
                {
                    case 1:
                        ImportSetOfImages( Mode.Training );
                        break;
                    case 2:
                        ImportSetOfImages( Mode.Testing );
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
                        SerializeCurrentHistory();
                        break;
                    case 9:
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
                Console.WriteLine("Will this be a number verifyers or recognizer");
                Console.WriteLine("Yes for ImageRecognizer");
                bool isImageRecognizer = Console.ReadLine() == "Yes" ? true : false;
                Console.WriteLine("Chose the Ammount of hidden Layers");
                bool inValidAmmount = true;
                while ( inValidAmmount )
                {
                    AmmountOfLayers = Convert.ToInt16( Console.ReadLine() );
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
                if( isImageRecognizer )
                {
                    Neurons.Add( 10 );
                }
                else
                {
                    Neurons.Add( 2 );
                }
                Console.WriteLine("Choose activation funtion \n 1 --> Sigmoid \n 2 --> Relu");
                bool inValidChoice = true;
                ActivationFunctionOptions activationChoice = ActivationFunctionOptions.ReLU;
                while( inValidChoice )
                {
                    int Choice = 0;
                    try
                    {
                        Choice = Convert.ToInt16( Console.ReadLine() );
                    }
                    catch
                    {
                        Console.WriteLine("Please enter a number");
                    }
                    if( Choice == 1 || Choice == 2 )
                    {
                        inValidChoice = false;
                        switch( Choice )
                        {
                            case 1: 
                                activationChoice = ActivationFunctionOptions.Sigmoid;
                                break;
                            case 2:
                                activationChoice = ActivationFunctionOptions.ReLU;
                                break;
                        }
                    }
                    else
                    {
                        Console.WriteLine("Please enter a valid ammount");
                    }
                }
                settings.SetCustom( AmmountOfLayers + 1, Neurons.ToArray(), activationChoice, isImageRecognizer );
            }
            else
            {
                settings.SetDefault();          
            }
            
            myManager.StartNewNetwork( settings );
        }
        private void SerializeCurrentHistory()
        {
            Console.WriteLine("Give the outputlocation");
            string? FileName = Console.ReadLine();
            if( FileName is not null )
            {
                myManager.SerializeCurrentHistory( FileName, true );
            }
        }
        private string GetJsonString()
        {
            Console.WriteLine("Give the Ouptut Folder");

            string? relativeOuptut = Console.ReadLine();
            if( relativeOuptut == null )
            {
                Console.WriteLine("No output received");
                return string.Empty;
            }
            else
            {
                string JsonString = string.Empty;
                try
                {
                    JsonString = File.ReadAllText( OutPuts.StandardJsonOutput + "" + relativeOuptut + "\\NetworkSettings.json");
                }
                catch( FileNotFoundException )
                {
                    Console.WriteLine("FileNotFound");
                }
                return JsonString;
            }

        }
        private void LoadInNetworkFromJson()
        {
            Console.WriteLine("Do you want the standard input");
            Console.WriteLine("( Yes or true )");
            string? Input = Console.ReadLine();
            bool StandarInput = ( Input == "true" ) || ( Input == "Yes" );
            Console.WriteLine("Would you like to load in a verifyer or a recognizer");
            Console.WriteLine("Yes for image Recognizer");
            bool iWantImageRecognizer = Console.ReadLine() == "Yes" ? true : false;
            if( StandarInput )
            {
                Console.WriteLine("Give the relative Ouptut Folder");
                bool OutputNotRecieved = true;
                while ( OutputNotRecieved )
                {
                    string JsonString = GetJsonString();
                    if( JsonString == string.Empty )
                    {

                    }
                    else
                    {
                        try
                        {
                            NetworkJsonFormat? JsonSettings = JsonSerializer.Deserialize<NetworkJsonFormat>( JsonString );
                            if( JsonSettings is null )
                            {
                                throw new NullReferenceException();
                            }
                            myManager.LoadInNetworkFromJson( JsonSettings, iWantImageRecognizer ); 
                            OutputNotRecieved = false;
                        }
                        catch( Exception )
                        {

                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("Give the absolut location of the .json file");
                bool OutputNotRecieved = true;
                while ( OutputNotRecieved )
                {
                    string? Location = GetJsonString();
                    if( Location != string.Empty )
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
                            OutputNotRecieved = false;
                        }
                        catch
                        {
                            Console.WriteLine("Something went wrong try again");
                        }
                    }
                    else
                    {
                        Console.WriteLine("No output received");
                    }
                }
            }
        }
        private void SerializeWheightAndBiasesToJson()
        {

            Console.WriteLine("Give the outputlocation");
            string? FileName = Console.ReadLine();
            Console.WriteLine("Serialize imageRecognizers -> Yes ");
            bool SerializeVerifyer = Console.ReadLine() == "Yes" ? true : false;
            if( FileName is not null )
            {
                myManager.SerializeWheightAndBiasesToJson( FileName, true, SerializeVerifyer );
            }
        }
        private void ImportSetOfImages( Mode mode )
        {
            Console.WriteLine("Do you wish to train the image recognizer of the image verifyer ");
            Console.WriteLine("ImageVerifyer for image verifyer");
            bool iamTrainingImageVerifyer = Console.ReadLine() == "ImageVerifyer" ? true : false;
            bool iamTrainingSpecificType = false;
            int TypeOfImageVerifyerToTrain = 0;
            if( iamTrainingImageVerifyer )
            {
                if( myManager.NumberVerifyerNetworks == null )
                {
                    return;
                }
                Console.WriteLine("Do you wish to train a specific type of image Verifyer");
                Console.WriteLine("Yes || any other input");
                iamTrainingSpecificType = Console.ReadLine() == "Yes" ? true : false;
                if( iamTrainingSpecificType )
                {
                    bool invalidinput = true;
                    while( invalidinput )
                    {
                        Console.WriteLine("Wich image verifyer would you like to train?");
                        try
                        {
                            TypeOfImageVerifyerToTrain = Convert.ToInt16( Console.ReadLine() );
                            if( TypeOfImageVerifyerToTrain >=  0 && TypeOfImageVerifyerToTrain < 10 )
                            {
                                invalidinput = false;
                            }
                            else
                            {
                                Console.WriteLine("Please enter a number between 1 and 10");
                            }
                        }
                        catch
                        {
                            Console.WriteLine("Invalid input");
                        }
                    }
                }
                for( int networkIndex = 0 ; networkIndex < 10 ; networkIndex++ )
                {
                    myManager.NumberVerifyerNetworks[ networkIndex ].displayBatchResults = DisplayBatchResults;
                    myManager.NumberVerifyerNetworks[ networkIndex ].displayResults = DisplayImageResults;
                }
            }
            else
            {
                if ( myManager.network is null )
                {
                    return;
                }
                myManager.network.displayBatchResults = DisplayBatchResults;
                myManager.network.displayResults = DisplayImageResults;
            }
            ImportSettings settings = new();

            Console.WriteLine("How Many images do you want to import?");
            settings.Ammount = Convert.ToInt16( Console.ReadLine() );
            Console.WriteLine("How Many Itteration do you want to run?");
            settings.Itterations = Convert.ToInt16( Console.ReadLine() );
            bool DisplayResults = false;

            Console.WriteLine("Do you want to display the results");
            Console.WriteLine("( Yes or true )");
            string? Input = Console.ReadLine();
            DisplayResults = ( Input == "true" ) || ( Input == "Yes" );

            Console.WriteLine("Do you want to add noise to the image");
            Console.WriteLine("( Yes or true )");
            string? NoisInput = Console.ReadLine();
            bool AddNoise = ( NoisInput == "true" ) || ( NoisInput == "Yes" );
            if( iamTrainingImageVerifyer )
            {
                if( iamTrainingSpecificType )
                {
                    myManager.TrainSpecificImageVerifyer( settings, mode, DisplayResults, AddNoise, TypeOfImageVerifyerToTrain );
                }
                else
                {
                    myManager.TrainImageVerifyer( settings, mode, DisplayResults, AddNoise );
                }
            }
            else
            {
                myManager.ImportSetOfImages( settings, mode, DisplayResults, AddNoise );
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


