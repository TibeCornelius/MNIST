using System.Text.Json;

namespace Ai.MNIST.NeuralNetworks
{
    public class ImageData( int ImageNumber, double Cost, int Guess, byte[,] Image, StNeuron[] OutputNeurons )
    {
        public int ImageNumber = ImageNumber;
        public byte[,] Image = Image;
        public int NumberGuessed = Guess;
        public double[] Results = [];
        public StNeuron[] neuronResults = OutputNeurons;
        public bool wasGuesCorrect => ImageNumber == NumberGuessed ;
        public double Cost = Cost; 
    }
    public class TrainingDataOutput
    {
        public List<ImageData> ImageData;
        public int CorrectGuesses;
        public double TotalAverageCost;
        public TrainingDataOutput()
        {
            this.ImageData = new List<ImageData>();
            this.CorrectGuesses = new int();
            this.TotalAverageCost = new double();
        }
    }

    public class Network
    {
        public List<Layer> NetworkLayers;
        public List<int> LiNetwork;
        public List<TrainingDataOutput> OurTrainingResults;
        public List<TrainingDataOutput> OurTestingResults;
        public Network( List<int> Network, bool New_Network = true, string _JsonFile  = "")
        {
            this.LiNetwork = Network;
            this.OurTrainingResults = new List<TrainingDataOutput>();
            this.OurTestingResults = new List<TrainingDataOutput>();
            
            this.NetworkLayers = new List<Layer>();
            if( New_Network )
            {
                int index = 0 ; 
                foreach( int layer in Network )
                {
                    NetworkLayers.Add( initialize_Layers( Network, index ) );
                    index++;
                }
            }
            else
            {
            initializeFromJson( Network, _JsonFile ); 
            }
        }

#region Json
        public void CreateJson( string OuputLocation )
        {
            Directory.CreateDirectory(".\\SavedSettings\\" + OuputLocation );
            foreach( Layer layer in NetworkLayers )
            {
                layer.CreateJson( OuputLocation );
            }
            CreateLayerCountJson( OuputLocation );
        }

        private void CreateLayerCountJson( string OuputLocation )
        {
            string LayerCountJsonString = JsonSerializer.Serialize( NetworkLayers.Count );
            File.WriteAllText( ".\\SavedSettings\\" + OuputLocation + "\\LayerCount.json", LayerCountJsonString );
        }

#endregion
#region initialization
        private Layer initialize_Layers( List<int> Network, int index )
        {
            Layer layer = new Layer( this, Network[ index ], index );
            return layer;
        }

        private void initializeFromJson( List<int> Network, string JsonFile )
        {
            int index = 0;
            foreach( int layer in Network )
            {
                NetworkLayers.Add( initialize_LayersFromJson( layer, index, JsonFile ) );
                index++;
            }
        }
        private Layer initialize_LayersFromJson( int layerNeuronCount, int index, string JsonFile )
        {
            bool NetworkFromJson = true;   
            Layer mylayer = new Layer( this, layerNeuronCount, index, NetworkFromJson, JsonFile );
            return mylayer;
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
        public TrainingDataOutput Train( List<byte[,]> images, List<string> labels, int TrainingSession )
        {
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingDataOutput myTraningResults = new TrainingDataOutput();
            for( int imageIndex = 0 ; imageIndex < images.Count ; imageIndex++ )
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
                myTraningResults.ImageData.Add( new ImageData( CorrectOutput, StImportedImage.cost, iGuessed, images[ imageIndex ], NetworkLayers[ NetworkLayers.Count - 1 ].StNeurons ) );
            }
            double TotalAverageCost = TotalCost( LiStImportedImages );
            double LearningRate = 0.001;
            ApplyAllGradients( LearningRate );
            ResetAllGradients();
            myTraningResults.CorrectGuesses = CorrectGuesses;
            myTraningResults.TotalAverageCost = TotalAverageCost;
            OurTrainingResults.Add( myTraningResults );
            return myTraningResults;
        }
        public TrainingDataOutput Test( List<byte[,]> images, List<string> labels, int TrainingSession )
        {
            List<ImportedImage> LiStImportedImages = new List<ImportedImage>();
            int CorrectGuesses = 0;
            TrainingDataOutput myResults = new TrainingDataOutput();
            for( int imageIndex = 0 ; imageIndex < images.Count ; imageIndex++ )
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
            OurTestingResults.Add( myResults );
            return myResults;
        }

        public TrainingDataOutput ImportSingleImage( Image image )
        {
            TrainingDataOutput results = new TrainingDataOutput();
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