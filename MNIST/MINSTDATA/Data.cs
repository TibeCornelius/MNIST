using Ai.MNIST.NeuralNetworks;


namespace Ai.MNIST.Data
{
    public struct Data( List<byte[,]> TrainingImages, List<string> TrainingLabels, List<byte[,]> TestingImages, List<string> TestingLabels )
    {
        private List<byte[,]> myTrainingImages =  TrainingImages;
        private List<string> myTrainingLabes = TrainingLabels;
        private List<byte[,]> myTestingImages = TestingImages;
        private List<string> myTestingLabels = TestingLabels;
        private Random myRandom = new Random();

        public Image GetSingleTrainingImage()
        {
            int index = myRandom.Next( 0, myTestingImages.Count );
            return new Image( myTrainingImages[ index ], myTrainingLabes[ index ] );
        }

        public Image GetSingleTestingImage()
        {
            int index = myRandom.Next( 0, myTestingImages.Count );
            return new Image( myTestingImages[ index ], myTestingLabels[ index ] );
        }

        public ToImportImages GetSetOfImages( int Ammount, Mode mode )
        {
            
            List<byte[,]> Images = mode switch
            {
                Mode.Training => myTrainingImages,
                Mode.Testing => myTestingImages,
                _ => throw new Exception(),
            };
            List<string> Labels = mode switch
            {
                Mode.Training => myTrainingLabes,
                Mode.Testing => myTestingLabels,
                _ => throw new Exception(),
            };
            int ListLenght = Images.Count;
            List<byte[,]> ToImport = new List<byte[,]>();
            List<string> stringToImport = new List<string>();
            for( int imageIndex = 0 ; imageIndex < Ammount ; imageIndex++ )
            {
                int index = myRandom.Next( 0, ListLenght );
                ToImport.Add( Images[ index ] );
                stringToImport.Add( Labels[ index ] );
            }
            return new ToImportImages( Ammount, ToImport, stringToImport );
        }

        
    }
}