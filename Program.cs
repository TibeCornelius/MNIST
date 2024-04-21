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
    public Program()
    {

    }

    public void something()
    {
        foreach (var image in MNIST.ReadTrainingData())
        {
            MNIST.ConvertToPng( image.Data );
            Console.WriteLine( Convert.ToInt16(image.Label) );
        }
    }


    public NeuralNetworks.CentralNeuralNetwork.Network initialize_CentralNeuralNetwork()
    {
        List<int> AmmountOfNeurons = [ 784, 400, 150, 10 ];
        NeuralNetworks.CentralNeuralNetwork.Network NeuralNetwork = new NeuralNetworks.CentralNeuralNetwork.Network( AmmountOfNeurons.Count, AmmountOfNeurons );
        return NeuralNetwork;
    }
    static void Main(string[] args)
    {
        Program Main = new Program();

        NeuralNetworks.CentralNeuralNetwork.Network NeuralNetwork = Main.initialize_CentralNeuralNetwork();

        foreach (var image in MNIST.ReadTrainingData())
        {
            NeuralNetwork.ImportImage( image.Data, Convert.ToString( image.Label ) );
        }

    }
}


