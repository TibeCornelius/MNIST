using System.Drawing;
using System.Drawing.Imaging;

namespace Ai.MNIST.Data
{
    public static class MNIST
    {
        /*private const string TrainImages = @"C:\Users\corne\Desktop\Everything\C#\HelloWorld\Ai\AiNumbers\DataSetMNIST\train-images.idx3-ubyte";
        private const string TrainLabels = @"C:\Users\corne\Desktop\Everything\C#\HelloWorld\Ai\AiNumbers\DataSetMNIST\train-labels.idx1-ubyte";
        private const string TestImages = @"C:\Users\corne\Desktop\Everything\C#\HelloWorld\Ai\AiNumbers\DataSetMNIST\train-images.idx3-ubyte";
        private const string TestLabels = @"C:\Users\corne\Desktop\Everything\C#\HelloWorld\Ai\AiNumbers\DataSetMNIST\t10k-labels.idx1-ubyte";*/
        private const string TrainImages = @"C:\Users\corne\Desktop\Everything\C#\Ai\AiMaui\Ai\MNIST\DataBase\train-images-idx3-ubyte\train-images-idx3-ubyte";
        private const string TrainLabels = @"C:\Users\corne\Desktop\Everything\C#\Ai\AiMaui\Ai\MNIST\DataBase\train-labels-idx1-ubyte\train-labels-idx1-ubyte";
        private const string TestImages = @"C:\Users\corne\Desktop\Everything\C#\Ai\AiMaui\Ai\MNIST\DataBase\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte";
        private const string TestLabels = @"C:\Users\corne\Desktop\Everything\C#\Ai\AiMaui\Ai\MNIST\DataBase\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte";
        enum Mode
        {
            Training,
            Testing
        }
        public static IEnumerable< Image > ReadTrainingData()
        {
            Mode mode = Mode.Training;
            foreach (var item in Read( TrainImages, TrainLabels, mode  ))
            {
                yield return item;
            }

        }

        public static IEnumerable< Image > ReadTestData()
        {
            Mode mode = Mode.Testing;
            foreach (var item in Read(TestImages, TestLabels, mode ))
            {
                yield return item;
            }

        }

        private static IEnumerable< Image > Read(string imagesPath, string labelsPath, Mode mode)
        {
            using ( BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open)))
            {
                using (BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open)))
                {
                    int ImageCount = 10000;
                    if( mode == Mode.Training )
                    {
                        ImageCount = 60000;
                    }
                    int magicNumber = images.ReadBigInt32();
                    int numberOfImages = images.ReadBigInt32();
                    int width = images.ReadBigInt32();
                    int height = images.ReadBigInt32();

                    int magicLabel = labels.ReadBigInt32();
                    int numberOfLabels = labels.ReadBigInt32();

                    for (int i = 0 ; i < ImageCount ; i++)
                    {
                        var bytes = images.ReadBytes(width * height);
                        var arr = new byte[height, width];

                        arr.ForEach((j,k) => arr[j, k] = bytes[j * height + k]);

                        yield return new Image()
                        {
                            Data = arr,
                            Label = labels.ReadByte()
                        };
                    }
                }
            }
        }
        public static float[,] ConvertByteTofloatArray( byte[,] image )
        {

           return new float[0,0]; 
        }
        /*public static void ConvertToPng( byte[,] image )
        {
            string PngOutput = @"C:\Users\corne\Desktop\Everything\C#\HelloWorld\Ai\AiNumbers\DataBase\PngFiles";
            int index = 0 ;
            // Create a directory if it doesn't exist
            Directory.CreateDirectory( PngOutput );

            // Generate a unique file name based on the current timestamp
            string fileName = $"image_{index}_{DateTime.Now:yyyyMMddHHmmssfff}.png";
            string outputPath = Path.Combine( PngOutput, fileName);

            // Create a new bitmap with the dimensions of an MNIST image (28x28)
#pragma warning disable CA1416 // Validate platform compatibility
            using (Bitmap bitmap = new Bitmap(28, 28))
            {
                // Set each pixel in the bitmap based on the corresponding value in the MNIST image data
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        int pixelValue = image[index + i , j ];
                        Color color = Color.FromArgb(pixelValue, pixelValue, pixelValue);
                        bitmap.SetPixel(j, i, color);
                    }
                }

                // Save the bitmap as a PNG image
                bitmap.Save(outputPath, ImageFormat.Png);
            }
#pragma warning restore CA1416 // Validate platform compatibility
        }*/
    }
    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }
    }
    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}