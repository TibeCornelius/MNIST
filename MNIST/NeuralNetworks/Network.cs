using System.Diagnostics;
using System.Text.Json;
using Ai.MNIST.NeuralNetworks.TrainingResults;
using Ai.MNIST.Util;

namespace Ai.MNIST.NeuralNetworks
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
        public Network( List<int> Network )
        {
            this.LiNetwork = Network; 
            this.NetworkLayers = new List<Layer>();
            this.OurResultsContainer = new Container();

            int index = 0 ; 
            foreach( int layer in Network )
            {
                NetworkLayers.Add( initialize_Layers( Network, index ) );
                index++;
            }
        }
        public Network( NetworkJsonFormat JsonSettings )
        {
            this.LiNetwork = new List<int>();
            this.NetworkLayers = new List<Layer>();
            this.OurResultsContainer = new Container();
            initializeFromJson( JsonSettings );
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
            NetworkJsonFormat jsonClass = new( NetworkLayers.Count, LiNetwork.ToArray(), GetAllWheights(), GetAllBiases() );
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
        private Layer initialize_Layers( List<int> Network, int index )
        {
            Layer layer = new Layer( this, Network[ index ], index );
            return layer;
        }

        private void initializeFromJson( NetworkJsonFormat JsonSettings )
        {
            LiNetwork = JsonSettings.NeuronCount.ToList();
            int index = 0; 
            foreach( int NeuronCount in LiNetwork )
            {
                Layer layer = new( this, Converter.JaggedToArray2D( JsonSettings.Weights[ index ] ), JsonSettings.Biases[ index ], index, JsonSettings.NeuronCount[ index ] );
                NetworkLayers.Add( layer );
                index++;
            }
        }

#endregion
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
        public TrainingBatch Train( ToImportImages ImportedImages, int TrainingSession )
        {
            Stopwatch stopWatch = new();
            stopWatch.Start();
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingBatch myTraningResults = new TrainingBatch( TrainingSession );
            List<byte[,]> allImages = ImportedImages.Images;
            List<string> allLabels = ImportedImages.Labels;
            for( int imageIndex = 0 ; imageIndex < allImages.Count ; imageIndex++ )
            {
                byte[,] ImageByte = allImages[ imageIndex ];
                string imageLabel = allLabels[ imageIndex ];
                ImportedImage StImportedImage = new ImportedImage
                {
                    image = ImageByte,
                    input = ImageByte,
                    output = ByteInput( ImageByte ),
                    excpectedOutput = CalculateCorrectOutputs( imageLabel ),
                    label = imageLabel
                };
                int CorrectOutput = Convert.ToInt16(StImportedImage.label);
                foreach ( Layer layer in NetworkLayers )
                {
                    layer.CalculateInputsEveryNeuron( StImportedImage );
                    layer.CalculateOutputs();
                }
                StImportedImage.cost = Cost( StImportedImage );
                int iGuessed = GetHighestOutput( StImportedImage );
                if( iGuessed == CorrectOutput )
                {
                    CorrectGuesses++;
                }

                Gradients( StImportedImage );
                LiStImportedImages.Add( StImportedImage );
                myTraningResults.ImageData.Add( new ImageData( CorrectOutput, StImportedImage.cost, iGuessed, ImageByte, NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons ) );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            double LearningRate = 0.001;
            ApplyAllGradients( LearningRate );
            ResetAllGradients();
            myTraningResults.CorrectGuesses = CorrectGuesses;
            myTraningResults.TotalAverageCost = TotalAverageCost;
            OurResultsContainer.OurTrainingResults.Add( myTraningResults );
            stopWatch.Stop();
            Console.WriteLine( stopWatch.Elapsed ); 
            return myTraningResults;
        }
        public TrainingBatch Train( ToImportImages ImportedImages, int TrainingSession , bool iwillDisplayResults )
        {
            
            if( displayResults is null || displayBatchResults is null )
            {
                throw new Exception();
            }
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingBatch myTraningResults = new TrainingBatch( TrainingSession );
            List<byte[,]> images = ImportedImages.Images;
            List<string> labels = ImportedImages.Labels;
            for( int imageIndex = 0 ; imageIndex < ImportedImages.Count ; imageIndex++ )
            {
                ImportedImage StImportedImage = new ImportedImage
                {
                    image = images[imageIndex],
                    input = images[imageIndex],
                    output = ByteInput(images[imageIndex]),
                    excpectedOutput = CalculateCorrectOutputs(labels[imageIndex]),
                    label = labels[imageIndex]
                };
                int CorrectOutput = Convert.ToInt16(StImportedImage.label);
                foreach ( Layer layer in NetworkLayers )
                {
                    layer.CalculateInputsEveryNeuron( StImportedImage );
                    layer.CalculateOutputs();
                }
                StImportedImage.cost = Cost( StImportedImage );
                int iGuessed = GetHighestOutput( StImportedImage );
                if( iGuessed == CorrectOutput )
                {
                    CorrectGuesses++;
                }

                Gradients( StImportedImage );

                LiStImportedImages.Add( StImportedImage );
                ImageData image = new ImageData( CorrectOutput, StImportedImage.cost, iGuessed, images[ imageIndex ], NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons );
                displayResults( image );
                myTraningResults.ImageData.Add(  image );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            double LearningRate = 0.001;
            ApplyAllGradients( LearningRate );
            ResetAllGradients();
            myTraningResults.CorrectGuesses = CorrectGuesses;
            myTraningResults.TotalAverageCost = TotalAverageCost;
            OurResultsContainer.OurTrainingResults.Add( myTraningResults );
            
            displayBatchResults( myTraningResults );
            return myTraningResults;
        }
        public TrainingBatch Test( ToImportImages ImportedImages, int TrainingSession )
        {
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingBatch myResults = new TrainingBatch( TrainingSession );
            List<byte[,]> images = ImportedImages.Images;
            List<string> labels = ImportedImages.Labels;
            for( int imageIndex = 0 ; imageIndex < ImportedImages.Count ; imageIndex++ )
            {
                ImportedImage StImportedImage = new ImportedImage
                {
                    image = images[imageIndex],
                    input = images[imageIndex],
                    output = ByteInput(images[imageIndex]),
                    excpectedOutput = CalculateCorrectOutputs(labels[imageIndex]),
                    label = labels[imageIndex]
                };
                int CorrectGues = Convert.ToInt16( StImportedImage.label );
                foreach ( Layer layer in NetworkLayers )
                {
                    layer.CalculateInputsEveryNeuron( StImportedImage );
                    layer.CalculateOutputs();
                }
                StImportedImage.cost = Cost( StImportedImage );
                int iGuessed = GetHighestOutput( StImportedImage );
                myResults.ImageData.Add( new ImageData( CorrectGues, StImportedImage.cost, iGuessed, images[ imageIndex ],  NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons ) );
                if( iGuessed == CorrectGues )
                {
                    CorrectGuesses++;
                }
                LiStImportedImages.Add( StImportedImage );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            myResults.CorrectGuesses = CorrectGuesses;
            myResults.TotalAverageCost = TotalAverageCost;
            OurResultsContainer.OurTestingResults.Add( myResults );
            return myResults;
        }
        public TrainingBatch Test( ToImportImages ImportedImages, int TrainingSession, bool iwillDisplayResults )
        {
            if( displayResults is null || displayBatchResults is null )
            {
                throw new Exception();
            }
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            List<byte[,]> images = ImportedImages.Images;
            List<string> labels = ImportedImages.Labels;
            TrainingBatch myResults = new TrainingBatch( TrainingSession );
            for( int imageIndex = 0 ; imageIndex < ImportedImages.Count ; imageIndex++ )
            {
                ImportedImage StImportedImage = new ImportedImage
                {
                    image = images[imageIndex],
                    input = images[imageIndex],
                    output = ByteInput(images[imageIndex]),
                    excpectedOutput = CalculateCorrectOutputs(labels[imageIndex]),
                    label = labels[imageIndex]
                };
                int CorrectGues = Convert.ToInt16( StImportedImage.label );
                foreach ( Layer layer in NetworkLayers )
                {
                    layer.CalculateInputsEveryNeuron( StImportedImage );
                    layer.CalculateOutputs();
                }
                StImportedImage.cost = Cost( StImportedImage );
                int iGuessed = GetHighestOutput( StImportedImage );
                ImageData image = new ImageData( CorrectGues, StImportedImage.cost, iGuessed, images[ imageIndex ],  NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons );
                myResults.ImageData.Add( image );
                displayResults( image );
                if( iGuessed == CorrectGues )
                {
                    CorrectGuesses++;
                }
                LiStImportedImages.Add( StImportedImage );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            myResults.CorrectGuesses = CorrectGuesses;
            myResults.TotalAverageCost = TotalAverageCost;
            displayBatchResults( myResults );
            OurResultsContainer.OurTestingResults.Add( myResults );
            return myResults;
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
                layer.ApplyGradients( LearningRate );
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

        public double[] CalculateCorrectOutputs( string label )
        {
            double[] output = new double[10];
            for( int index = 0 ; index < 10 ; index++ )
            {
                output[ index ] = ( index == Convert.ToInt16( label ) ) ? 1 : 0 ;
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
            double[] outputs = new double[ 10 ];
            int index = 0;
            Layer outputLayer = NetworkLayers[ NetworkLayers.Count - 1 ];
            foreach ( StNeuron neuron in NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons )
            {
                outputs[ index ] = outputLayer.StNeurons[0].input; 
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
            double[] outputs = new double[ 10 ];
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