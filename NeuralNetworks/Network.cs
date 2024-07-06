using System.Diagnostics;
using System.Text.Json;
using MNIST.NeuralNetworks.TrainingResults;
using MNIST.Util;

namespace MNIST.NeuralNetworks
{
    public sealed class Network
    {
        public List<Layer> NetworkLayers;
        public List<int> LiNetwork;
        public Container OurResultsContainer;
        public delegate void DisplayImageResults( ImageData image );
        public DisplayImageResults? displayResults{ get; set; }
        public delegate void DisplayBatchResults( TrainingBatch trainingBatch );
        public DisplayBatchResults? displayBatchResults{ get; set; }
        private ActivationFunctionOptions myActivationFunction;
        private bool iamImageRecognizer;
        private int myNumberToRecognize;
        private int myOutputNeurons;
        private Manager myManager;
        private const double HyperParameterTuner = 0.5;//Used for momentum base gradient descent
        public Network( List<int> Network, ActivationFunctionOptions activationFunctionOptions, bool iamImageRecognizer, int myNumberToRecognize, Manager manager )
        {
            this.LiNetwork = Network; 
            this.NetworkLayers = new List<Layer>();
            this.OurResultsContainer = new Container();
            this.iamImageRecognizer = iamImageRecognizer;
            this.myNumberToRecognize = myNumberToRecognize;
            this.myOutputNeurons = iamImageRecognizer ? 10 : 2;
            this.myManager = manager;

            int index = 0 ; 
            foreach( int layer in Network )
            {
                NetworkLayers.Add( initialize_Layers( Network, index, activationFunctionOptions ) );
                index++;
            }
        }
        public Network( NetworkJsonFormat JsonSettings, Manager manager )
        {
            this.LiNetwork = new List<int>();
            this.NetworkLayers = new List<Layer>();
            this.OurResultsContainer = new Container();
            this.myManager = manager;
            initializeFromJson( JsonSettings );
            if( JsonSettings.iamImageRecognizer == false )
            {
                this.myNumberToRecognize = JsonSettings.myImageToRecognize;
            }
            this.iamImageRecognizer = JsonSettings.iamImageRecognizer;
        }
#region TrainingStats
        public void LoadInStatsFromJson()
        {

        }
#endregion
#region Json
        public void SerializeStatsToJson( string OutputLocation, bool DefaultLocation )
        {
            if( DefaultLocation )
            {
                Directory.CreateDirectory(".\\SavedStats\\" + OutputLocation );
                string JsonString = JsonSerializer.Serialize( OurResultsContainer );
                File.WriteAllText( ".\\SavedStats\\" + OutputLocation + "\\Stats.json", JsonString );
            }
            else
            {
                Directory.CreateDirectory( OutputLocation );
                string JsonString = JsonSerializer.Serialize( OurResultsContainer );
                File.WriteAllText( OutputLocation + "\\Stats.json", JsonString );
            }
        }
        public string CreateJson( string OuputLocation, bool DefaultLocation )
        {
            int imageToRecognize = iamImageRecognizer ? 0 : myNumberToRecognize;
            NetworkJsonFormat jsonClass = new( NetworkLayers.Count, LiNetwork.ToArray(), GetAllWheights(), GetAllBiases(), myActivationFunction, iamImageRecognizer,imageToRecognize );
            string NetworkJson = JsonSerializer.Serialize( jsonClass );
            if( DefaultLocation )
            {
                Directory.CreateDirectory(".\\SavedSettings\\" + OuputLocation );
                File.WriteAllText( ".\\SavedSettings\\" + OuputLocation + "\\NetworkSettings.json", NetworkJson );
            }
            else
            {
                Directory.CreateDirectory( OuputLocation );
                File.WriteAllText( OuputLocation + "\\NetworkSettings.json", NetworkJson );
            }
            return NetworkJson; 
        }
        public List<double[]> GetAllBiases()
        {
            List<double[]> AllBiases = new();
            foreach( Layer layer in NetworkLayers )
            {
                AllBiases.Add( layer.GetBiasesToArray() );
            }
            return AllBiases;
        }
        public List<double[,]> GetAllWheights()
        {
            List<double[,]> AllWheights = new();
            foreach( Layer layer in NetworkLayers )
            {
                AllWheights.Add( layer.WeightsPreviousLayer );
            }
            return AllWheights;
        }
#endregion
#region initialization
        private Layer initialize_Layers( List<int> Network, int index,ActivationFunctionOptions activationFunctionOptions )
        {
            Layer layer = new Layer( this, Network[ index ], index, activationFunctionOptions );
            return layer;
        }
        private void initializeFromJson( NetworkJsonFormat JsonSettings )
        {
            LiNetwork = JsonSettings.NeuronCount.ToList();
            int index = 0; 
            foreach( int NeuronCount in LiNetwork )
            {
                bool LastLayer = index == LiNetwork.Count - 1 ? true : false;
                Layer layer = new( this, Converter.JaggedToArray2D( JsonSettings.Weights[ index ] ), JsonSettings.Biases[ index ], index, JsonSettings.NeuronCount[ index ], LastLayer, ActivationFunctionOptions.Sigmoid );
                NetworkLayers.Add( layer );
                index++;
            }
        }

#endregion
        public RefStNeuron GetNeuronsPrevLayer( int Layer )
        {
            return new RefStNeuron( ref NetworkLayers[ Layer - 1 ].StNeurons );
        }
        public double[] ByteInput( byte[,] image )
        {
            double[] output = new double[ 28 * 28 ];
            for( int row = 0, total = 0 ; row < 28 ; row++ )
            {
                for( int column = 0 ; column < 28 ; column++, total++ )
                {
                    output[ row * 28 + column ] = ByteToNeuronInput( image[ row, column ] );
                }
            }
            return output;
        }
        private double ByteToNeuronInput( double WeightedInput )
        {
            double returnvalue = WeightedInput / 255;
            return returnvalue;
        }
        public TrainingBatch RunImagesThroughNetwork( ToImportImages ImportedImages, bool TrainNetwork , int Session, bool iwillDisplayResults, bool VerifyResults )
        {
            if( iwillDisplayResults && ( displayResults is null || displayBatchResults is null ) )
            {
                throw new NullReferenceException();
            }
            Stopwatch stopWatch = new();
            stopWatch.Start();
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingBatch myTraningResults = new TrainingBatch( Session );
            List<byte[,]> allImages = ImportedImages.Images;
            List<int> allLabels = ImportedImages.Labels;
            for( int imageIndex = 0 ; imageIndex < allImages.Count ; imageIndex++ )
            {
                byte[,] ImageByte = allImages[ imageIndex ];
                int imageLabel = allLabels[ imageIndex ];
                ImportedImage StImportedImage = new ImportedImage
                {
                    image = ImageByte,
                    input = ImageByte,
                    output = ByteInput( ImageByte ),
                    excpectedOutput = CalculateCorrectOutputs( imageLabel ),
                    label = imageLabel
                };
                foreach ( Layer layer in NetworkLayers )
                {
                    layer.CalculateInputsEveryNeuron( StImportedImage );
                    layer.CalculateOutputs();
                }
                StImportedImage.cost = Cost( StImportedImage );
                int iGuessed = GetHighestOutput( StImportedImage );
                if( iamImageRecognizer )
                {  
                    if( VerifyResults )
                    {
                        int index = 0;
                        do
                        {
                            bool VerificaitonResult = myManager.VerifyNetworkOutputs( new Image( ImageByte, imageLabel ), iGuessed  ); 
                            iGuessed = GetXHighestOutput( StImportedImage, iGuessed );
                            index++;
                        }
                        while( VerifyResults == false && index < 10 );
                    }
                    if( iGuessed == StImportedImage.label )
                    {
                        CorrectGuesses++;
                    }
                }
                else
                {
                    if( imageLabel == myNumberToRecognize && iGuessed == 1 )
                    {
                        CorrectGuesses++;
                    }
                }

                Gradients( StImportedImage );
                LiStImportedImages.Add( StImportedImage );
                ImageData image = new ImageData( StImportedImage.label, StImportedImage.cost, iGuessed, ImageByte, NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons );
                if( iwillDisplayResults )
                {
#pragma warning disable CS8602// Dereference of a possibly null reference.
                    displayResults( image );
#pragma warning restore CS8602// Dereference of a possibly null reference.
                }
                myTraningResults.ImageData.Add( image );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            double LearningRate = 0.001;
            if( TrainNetwork )
            {
                ApplyAllGradients( LearningRate );
                ResetAllGradients();
            }
            myTraningResults.CorrectGuesses = CorrectGuesses;
            myTraningResults.TotalAverageCost = TotalAverageCost;
            OurResultsContainer.OurTrainingResults.Add( myTraningResults );
            if( iwillDisplayResults )
            {
#pragma warning disable CS8602 // Dereference of a possibly null reference.
                displayBatchResults( myTraningResults );
#pragma warning restore CS8602 // Dereference of a possibly null reference.
            }
            stopWatch.Stop();
            Console.WriteLine( stopWatch.Elapsed ); 
            return myTraningResults;
        }

        internal TrainingBatch ImportSingleImage( Image image )
        {
            TrainingBatch results = new TrainingBatch( 1 );
            ImportedImage StImportedImage = new ImportedImage
            {
                image = image.ImageData,
                input = image.ImageData,
                output = ByteInput( image.ImageData ),
                excpectedOutput = CalculateCorrectOutputs( image.Label ),
                label = image.Label,
            };
            int CorrectOutput = Convert.ToInt16( image.Label );
            foreach( Layer layer in NetworkLayers )
            {
                layer.CalculateInputsEveryNeuron( StImportedImage );
                layer.CalculateOutputs();
            }
            StImportedImage.cost = Cost( StImportedImage );
            int iGuessed = GetHighestOutput( StImportedImage );
            results.ImageData.Add( new ImageData( CorrectOutput, StImportedImage.cost, iGuessed, image.ImageData,  NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons  ) );
            return results;
        }   
        private void ResetAllGradients()
        {
            foreach( Layer layer in NetworkLayers )
            {
                layer.ResetGradients();
            }
        }
        private void ApplyAllGradients( double LearningRate )
        {
            foreach( Layer layer in NetworkLayers )
            {
                layer.ApplyGradients( LearningRate, HyperParameterTuner );
            }
        }
        private void Gradients( ImportedImage StImportedImage )
        {   
            Layer outputLayer = NetworkLayers[ NetworkLayers.Count - 1 ];
            double[] nodeValues = outputLayer.NodeValuesFinalLayer( StImportedImage.excpectedOutput );
            NetworkLayers[ NetworkLayers.Count - 1 ].UpdateGradient( nodeValues );

            for( int layerIndex = NetworkLayers.Count - 2 ; layerIndex >= 0 ; layerIndex-- )
            {
                Layer hiddenLayer = NetworkLayers[ layerIndex ];
                nodeValues = hiddenLayer.CalculateHiddenLayersNodeValues( NetworkLayers[ layerIndex + 1 ], nodeValues );
                if( layerIndex > 0 )
                {
                    hiddenLayer.UpdateGradient( nodeValues );
                }
                else
                {
                    hiddenLayer.UpdateGradientLastLayer( nodeValues, StImportedImage );
                }
            }
        }
        private int GetHighestOutput( ImportedImage StImportedImage )
        {
            double output = new double();
            int iOutput = 0;
            int index = 0;
            foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
            {
                if( neuron.output > output )
                {
                    output = neuron.output;
                    iOutput = index;
                }
                index++;
            }
            return iOutput;
        }
        private int GetXHighestOutput( ImportedImage StImportedImage, int LastHighest )
        {
            double output = new double();
            int iOutput = 0;
            int index = 0;

            foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
            {
                if( neuron.output > output && neuron.output < LastHighest )
                {
                    output = neuron.output;
                    iOutput = index;
                }
                index++;
            }
            return iOutput;
        }
        public double[] CalculateCorrectOutputs( int label )
        {
            double[] output = new double[ myOutputNeurons ];
            if( iamImageRecognizer )
            {
                for( int index = 0 ; index < 10 ; index++ )
                {
                    output[ index ] = ( index == label ) ? 1 : 0 ;
                }
            }
            else
            {
                for( int index = 0 ; index < 2 ; index++ )
                {
                    if( index == 0 )
                    {
                        output[ index ] =  label == myNumberToRecognize ? 0 : 1;  
                    }
                    else
                    {
                        output[ index ] =  label == myNumberToRecognize ? 1: 0;  
                    }
                }
            }

            return output;
        }
        public double TotalCost( List< ImportedImage > LiStImportedImages )
        {
            double totalcost = new double();
            foreach( ImportedImage importedImage in LiStImportedImages )
            {
                totalcost += importedImage.cost;
            }
            return totalcost / LiStImportedImages.Count;
        }
        public double Nodecost( double outPutActivation, double ExpectedOutput )
        {
            double error = outPutActivation - ExpectedOutput;
            return error * error ;
        }  
        public double NodeCostDerrivative( double outPutActivation, double ExpectedOutput )
        {
            return 2 * ( outPutActivation - ExpectedOutput );
        }
        public double CostDerrivative( ImportedImage StImportedImage )
        {
            double[] outputs = new double[ myOutputNeurons ];
            int index = 0;
            Layer outputLayer = NetworkLayers[ NetworkLayers.Count - 1 ];
            foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
            {
                outputs[ index ] = neuron.input; 
            }
            double[] ExcpectedOutput = CalculateCorrectOutputs( StImportedImage.label );
            double cost = 0;

            for( int nodeOut = 0 ; nodeOut < outputs.Length ; nodeOut++ )
            {
                cost += NodeCostDerrivative( outputs[ nodeOut ], ExcpectedOutput[ nodeOut ] );
            }
            cost = cost / outputs.Length;

            return cost;
        }
        private double Cost( ImportedImage StImportedImage )
        {
            double[] outputs = new double[ myOutputNeurons ];
            int index = 0;
            Layer outputLayer = NetworkLayers[ NetworkLayers.Count - 1 ];
            foreach ( StNeuron neuron in outputLayer.StNeurons )
            {
                outputs[ index ] = neuron.output;
                index++; 
            }
            double[] ExcpectedOutput = CalculateCorrectOutputs( StImportedImage.label );
            double cost = 0;

            for( int nodeOut = 0 ; nodeOut < outputs.Length ; nodeOut++ )
            {
                cost += Nodecost( outputs[ nodeOut ], ExcpectedOutput[ nodeOut ] );
            }
            cost = cost / outputs.Length;
            return cost;
        }


    }
}