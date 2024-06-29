using Ai.MNIST.NeuralNetworks;


namespace Ai.MNIST.Data
{
    public struct Data( List<byte[,]> TrainingImages, List<int> TrainingLabels, List<byte[,]> TestingImages, List<int> TestingLabels )
    {
        private List<byte[,]> myTrainingImages =  TrainingImages;
        private List<int> myTrainingLabes = TrainingLabels;
        private List<byte[,]> myTestingImages = TestingImages;
        private List<int> myTestingLabels = TestingLabels;
        private Random myRandom = new Random();

        public Image GetSingleTrainingImage( bool AddNoise = false )
        {
            int index = myRandom.Next( 0, myTestingImages.Count );
            return AddNoise ? 
            new Image( AddNoiseToImage( myTrainingImages[ index ] ), myTrainingLabes[ index ] ): 
            new Image( myTrainingImages[ index ], myTrainingLabes[ index ] );
        }

        public Image GetSingleTestingImage( bool AddNoise = false )
        {
            int index = myRandom.Next( 0, myTestingImages.Count );
            return AddNoise ?
            new Image( AddNoiseToImage(myTestingImages[ index ] ), myTestingLabels[ index ] ):
            new Image( myTestingImages[ index ], myTestingLabels[ index ] );
        }

        public byte[,] AddNoiseToImage( byte[,] image )
        {
            for (int row = 0; row < image.GetLength( 0 ); row++)
            {
                for (int column = 0; column < image.GetLength( 1 ); column++)
                {
                    if ( image[ row, column ] < 200)
                    {
                        if ( myRandom.NextDouble() < 0.1)
                        {
                            int noise = myRandom.Next( 0, ( 256 - image[row, column] ) / 2 );
                            image[ row, column ] = ( byte )Math.Min( 255, image[ row, column ] + noise );
                        }
                    }
                }
            }
            return image;
        }

        public ToImportImages GetSetOfImages( int Ammount, Mode mode, bool AddNoise = false )
        {
            
            List<byte[,]> Images = mode switch
            {
                Mode.Training => myTrainingImages,
                Mode.Testing => myTestingImages,
                _ => throw new Exception(),
            };
            List<int> Labels = mode switch
            {
                Mode.Training => myTrainingLabes,
                Mode.Testing => myTestingLabels,
                _ => throw new Exception(),
            };
            int ListLenght = Images.Count;
            List<byte[,]> ToImport = new List<byte[,]>();
            List<int> iToImport = new List<int>();
            for( int imageIndex = 0 ; imageIndex < Ammount ; imageIndex++ )
            {
                int index = myRandom.Next( 0, ListLenght );
                if( AddNoise )
                {
                    byte[,] image = AddNoiseToImage( Images[ index ] );
                    ToImport.Add( image );
                }
                else
                {
                    ToImport.Add( Images[ index ] );
                }
                iToImport.Add( Labels[ index ] );
            }
            return new ToImportImages( Ammount, ToImport, iToImport );
        }

        
    }
}