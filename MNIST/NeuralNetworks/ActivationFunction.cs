using Google.Protobuf.WellKnownTypes;

namespace Ai.MNIST.NeuralNetworks
{
    public static class ActivationFunctions
    {
        public static double Sigmoid( double WeightedInput )
        {
            double output = 1 / ( 1 + Math.Exp( -WeightedInput));

            return output;
        }
        public static double SigmoidDx( double WeightedInput )
        {
            double Activation = Sigmoid( WeightedInput );
            return Activation * ( 1 - Activation );
        }
        public static double ReLU( double WeightedInput )
        {
            return WeightedInput > 0 ? WeightedInput : 0.01 * WeightedInput;
        }
        public static double ReLUDx( double WeightedInput )
        {
            return WeightedInput > 0 ? 1 : 001;
        }
        public static double[] SoftMax( double[] WeightedInputs )
        {
            double maxInputsValue = WeightedInputs.Max();
            double[] ExponentiatedValues = WeightedInputs.Select( value => Math.Exp( value - maxInputsValue ) ).ToArray();
            double sumOfExponentiatedValues = ExponentiatedValues.Sum();
            return ExponentiatedValues.Select( value => value / sumOfExponentiatedValues ).ToArray();
        }

        public static double NextGaussian( this Random rand, double mean = 0, double stddev = 1)
        {
            // Box-Muller transform
            double u1 = 1.0 - rand.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Random normal(0,1)
            return mean + stddev * randStdNormal; // Random normal(mean,stdDev^2)
        }


    }
}