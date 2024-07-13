namespace MNIST.Util
{
    public class ZoomMatrix
    {
        private double myZoomFactor;
        private double[,] myMatrix;
        private int myFocalPoint;
        public ZoomMatrix( double zoomFactor, int FocalPoint )
        {
            this.myZoomFactor = zoomFactor;
            this.myMatrix = new double[2,2]
            {
                {zoomFactor, 0 },
                { 0, zoomFactor }
            };
            this.myFocalPoint = FocalPoint;
        }

        public double[] GetXandY( double x , double y )
        {
            double newX = x * myZoomFactor;
            double newY = y * myZoomFactor;
            return [ newX, newY ];
        }
    }
}