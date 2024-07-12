using MNIST.NeuralNetworks;
using MNIST.Util;


namespace MNIST.Data
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
        public static byte[,] InvertColors( byte[,] image )
        {
            int rowLength = image.GetLength( 0 );
            int columnLength = image.GetLength( 1 );
            for( int rowIndex = 0 ; rowIndex < rowLength ; rowIndex++ )
            {
                for( int columnIndex = 0 ; columnIndex < columnLength ; columnIndex++ )
                {
                    int PixelValue = image[ rowIndex, columnIndex ];
                    image[ rowIndex, columnIndex ] = (byte)Math.Abs( PixelValue - 255 );
                }
            }
            return image;
        }

        public static byte[,] RotateImage(byte[,] originalImage, float angle)
        {
            RotationMatrix rotationMatrix = new RotationMatrix(angle);

            int height = originalImage.GetLength(0);
            int width = originalImage.GetLength(1);
            byte[,] rotatedImage = new byte[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float[] newCoords = rotationMatrix.GetXandY(x - width / 2.0f, y - height / 2.0f);
                    newCoords[0] += width / 2.0f;
                    newCoords[1] += height / 2.0f;

                    int xFloor = (int)Math.Floor(newCoords[0]);
                    int xCeiling = (int)Math.Ceiling(newCoords[0]);
                    int yFloor = (int)Math.Floor(newCoords[1]);
                    int yCeiling = (int)Math.Ceiling(newCoords[1]);

                    double xDecimals = newCoords[0] - xFloor;
                    double yDecimals = newCoords[1] - yFloor;

                    if (xFloor >= 0 && xFloor < width && yFloor >= 0 && yFloor < height)
                    {
                        double Value = originalImage[ y, x ] * ( 1 - xDecimals ) * ( 1 - yDecimals );
                        double RoundedValue = Value > 255 ? 255 : Value; 
                        rotatedImage[ yFloor, xFloor ] += (byte)RoundedValue;
                    }
                    if (xCeiling >= 0 && xCeiling < width && yFloor >= 0 && yFloor < height)
                    {
                        double Value = originalImage[ y, x ] * xDecimals * ( 1 - yDecimals );
                        double RoundedValue = Value > 255 ? 255 : Value; 
                        rotatedImage[ yFloor, xCeiling ] += (byte)RoundedValue;
                    }
                    if (xFloor >= 0 && xFloor < width && yCeiling >= 0 && yCeiling < height)
                    {
                        double Value = originalImage[ y, x ] * ( 1 - xDecimals ) * yDecimals ;
                        double RoundedValue = Value > 255 ? 255 : Value; 
                        rotatedImage[ yCeiling, xFloor] += (byte)RoundedValue;
                    }
                    if (xCeiling >= 0 && xCeiling < width && yCeiling >= 0 && yCeiling < height)
                    {
                        double Value = originalImage[ y, x ] * xDecimals * yDecimals;
                        double RoundedValue = Value > 255 ? 255 : Value; 
                        rotatedImage[ yCeiling, xCeiling ] += (byte)RoundedValue;
                    }
                }
            }

            return rotatedImage;
        }

        public static byte[,] ZoomImage( byte[,] originalImage, float zoomFactor, int focalX, int focalY)
        {
            int originalWidth = originalImage.GetLength(1);
            int originalHeight = originalImage.GetLength(0);

            int zoomedWidth = (int)(originalWidth * zoomFactor);
            int zoomedHeight = (int)(originalHeight * zoomFactor);

            int newArrayHeight = zoomedWidth > 28 ? zoomedWidth : 28;
            int newArrayWidth = zoomedWidth > 28 ? zoomedWidth : 28;
            byte[,] zoomedImage = new byte[ newArrayHeight, newArrayWidth ];

            // Calculate the center of the zoomed image
            //float centerX = focalX * zoomFactor;
            //float centerY = focalY * zoomFactor;
            float centerX = focalX;
            float centerY = focalY;
            int HeightEdge = zoomFactor > 1 ? originalHeight : zoomedHeight;
            int WidthEdge = zoomFactor > 1 ? originalWidth : zoomedWidth;
            for (int y = 0; y < HeightEdge; y++)
            {
                for (int x = 0; x < WidthEdge; x++)
                {
                    // Calculate the original pixel positions relative to the focal point
                    float origX = (x - centerX) / zoomFactor + focalX;
                    float origY = (y - centerY) / zoomFactor + focalY;

                    if (origX >= 0 && origX < originalWidth && origY >= 0 && origY < originalHeight)
                    {
                        // Get the coordinates of the four surrounding pixels
                        int x1 = (int)Math.Floor(origX);
                        int y1 = (int)Math.Floor(origY);
                        int x2 = Math.Min(x1 + 1, originalWidth - 1);
                        int y2 = Math.Min(y1 + 1, originalHeight - 1);

                        // Calculate the distances from the original pixel
                        float xFraction = origX - x1;
                        float yFraction = origY - y1;

                        // Perform bilinear interpolation
                        byte topLeft = originalImage[y1, x1];
                        byte topRight = originalImage[y1, x2];
                        byte bottomLeft = originalImage[y2, x1];
                        byte bottomRight = originalImage[y2, x2];

                        byte topInterpolated = (byte)(topLeft + xFraction * (topRight - topLeft));
                        byte bottomInterpolated = (byte)(bottomLeft + xFraction * (bottomRight - bottomLeft));
                        byte finalValue = (byte)(topInterpolated + yFraction * (bottomInterpolated - topInterpolated));

                        zoomedImage[y, x] = finalValue;
                    }
                    else
                    {
                        zoomedImage[y, x] = 0; // Fill with black if outside bounds
                    }
                }
            }

            return zoomedImage;
        }

        public static byte[,] AddNoiseToImage( byte[,] image )
        {
            Parallel.For( 0, image.GetLength( 0 ), row => 
            {
                Random random = new();
                for (int column = 0; column < image.GetLength( 1 ); column++)
                {
                    if ( image[ row, column ] < 200)
                    {
                        if ( random.NextDouble() < 0.1)
                        {
                            int noise = random.Next( 0, ( 256 - image[row, column] ) / 2 );
                            image[ row, column ] = ( byte )Math.Min( 255, image[ row, column ] + noise );
                        }
                    }
                }
            });
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