namespace Ai.MNIST.Util
{
    public static class Converter
    {
        public static T[][] Array2DToJagged<T>(T[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            T[][] jaggedArray = new T[rows][];
            for (int i = 0; i < rows; i++)
            {
                jaggedArray[i] = new T[cols];
                for (int j = 0; j < cols; j++)
                {
                    jaggedArray[i][j] = array[i, j];
                }
            }
            return jaggedArray;
        }
        public static T[,] JaggedToArray2D<T>(T[][] jaggedArray)
        {
            int rows = jaggedArray.Length;
            int cols = jaggedArray[0].Length;
            T[,] array2D = new T[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    array2D[i, j] = jaggedArray[i][j];
                }
            }
            return array2D;
        }
        public static T[][][] List2DArrayToJaggedArray<T>(List<T[,]> list)
        {
            return list.Select(array =>
            {
                int rows = array.GetLength(0);
                int cols = array.GetLength(1);
                T[][] jaggedArray = new T[rows][];
                for (int i = 0; i < rows; i++)
                {
                    jaggedArray[i] = new T[cols];
                    for (int j = 0; j < cols; j++)
                    {
                        jaggedArray[i][j] = array[i, j];
                    }
                }
                return jaggedArray;
            }).ToArray();
        }
        public static List<T[,]> Jagged3DArrayToList2DArray<T>(T[][][] jaggedArray)
        {
            return jaggedArray.Select(array =>
            {
                int rows = array.Length;
                int cols = array[0].Length;
                T[,] multiArray = new T[rows, cols];
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        multiArray[i, j] = array[i][j];
                    }
                }
                return multiArray;
            }).ToList();
        }
        public static T[][] ListToJaggedArray<T>(List<T[]> list)
        {
            return list.ToArray();
        }
        public static List<T[]> JaggedArrayToList<T>(T[][] jaggedArray)
        {
            return jaggedArray.ToList();
        }
    }
}