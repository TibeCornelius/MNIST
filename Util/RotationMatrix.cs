
namespace MNIST.Util
{
    public class RotationMatrix
    {
        private float myAngle;
        public RotationMatrix( float angle )
        {
            this.myAngle = (float)( angle * Math.PI / 180 );
        }

        public float[] GetXandY( float x, float y )
        {
            float newX = (float)(x * Math.Cos( (double)myAngle ) - y * Math.Sin( (double)myAngle));
            float newY = (float)(x * Math.Sin( (double)myAngle ) + y * Math.Cos( (double)myAngle));
            return [ newX, newY ];
        }
        
    }
}