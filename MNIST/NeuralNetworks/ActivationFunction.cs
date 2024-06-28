using Google.Protobuf.WellKnownTypes;

namespace Ai.MNIST.NeuralNetworks
{
    public static class ActivationFunction
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
            return Math.Max( 0, WeightedInput );
        }
        public static double[] SoftMax( double[] WeightedInputs )
        {
            double maxInputsValue = WeightedInputs.Max();
            double[] ExponentiatedValues = WeightedInputs.Select( value => Math.Exp( value - maxInputsValue ) ).ToArray();
            double sumOfExponentiatedValues = ExponentiatedValues.Sum();
            return ExponentiatedValues.Select( value => value / sumOfExponentiatedValues ).ToArray();
        }


    }
}