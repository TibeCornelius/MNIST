using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using MNISTDATA;
using Tensorflow;
using Tensorflow.Data;
using Tensorflow.NumPy;
using Tensorflow.Training;

class Program
{
    NeuralNetworks.Latest.Network network;
    public Program( List<int> Neurons )
    {
        this.network = new NeuralNetworks.Latest.Network( Neurons );
    }

    //public void something()
    //{
    //    foreach (var image in MNIST.ReadTrainingData())
    //    {
    //        MNIST.ConvertToPng( image.Data );
    //        Console.WriteLine( Convert.ToInt16(image.Label) );
    //    }
    //}


    //public NeuralNetworks.CentralNeuralNetwork.Network initialize_CentralNeuralNetwork()
    //{
    //    List<int> AmmountOfNeurons = [ 784, 400, 150, 10 ];
    //    NeuralNetworks.CentralNeuralNetwork.Network NeuralNetwork = new NeuralNetworks.CentralNeuralNetwork.Network( AmmountOfNeurons.Count, AmmountOfNeurons );
    //    return NeuralNetwork;
    //}

    public void ImportSetOfImages()
    {
        MNISTDATA.Image DataSet = new MNISTDATA.Image();
        List< byte[,] > list = new List< byte[,] >();
        List< string > sList = new List<string>();

        foreach( MNISTDATA.Image image in MNIST.ReadTrainingData() )
        {
            list.Add( image.Data );
            sList.Add( Convert.ToString( image.Label ) );
        }
        Random random = new Random();
        while( true )
        {
            List< byte[,] > listToTrain = new List< byte[,] >();
            List< string > sListToTrain = new List<string>();
            
            for( int index = 0 ; index < 100 ; index++ )
            {
                int randomnumber = random.Next( list.Count );
                listToTrain.Add( list[ randomnumber ] );
                sListToTrain.Add( sList[ randomnumber ] );
            }
            network.Train( listToTrain, sListToTrain );
        }
    }

    public void ImportImage()
    {    
        foreach( MNISTDATA.Image image in MNIST.ReadTrainingData() )
        {
            List< byte[,] > list = new List< byte[,] >();
            list.Add( image.Data );
            List< string > sList = new List<string>();
            sList.Add( Convert.ToString( image.Label ) );
            network.Train( list, sList );
        }
    }
    static void Main(string[] args)
    {
        List<int> AmmountOfNeurons = [ 400, 150, 10 ];
        //List<int> AmmountOfNeurons = [ 5, 10 ];
        Program Main = new Program( AmmountOfNeurons );
        Main.ImportSetOfImages();

        //NeuralNetworks.CentralNeuralNetwork.Network NeuralNetwork = Main.initialize_CentralNeuralNetwork();

        //foreach (var image in MNIST.ReadTrainingData())
        //{
        //    NeuralNetwork.ImportImage( image.Data, Convert.ToString( image.Label ) );
        //}

    }
}


