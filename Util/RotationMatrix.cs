
namespace MNIST.Util
{
    public class RotationMatrix
    {
        private double myAngle;
        public RotationMatrix( double angle )
        {
            this.myAngle =  angle * Math.PI / 180;
        }

        public double[] GetXandY( double x, double y )
        {
            double newX = x * Math.Cos( myAngle ) - y * Math.Sin( myAngle);
            double newY = x * Math.Sin( myAngle ) + y * Math.Cos( myAngle);
            return [ newX, newY ];
        }
        
    }
}