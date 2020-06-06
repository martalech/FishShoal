using Alea;

namespace FishShoal
{
    public static class FishGPU
    {
        public static void Kernel(double[] positionsx, double[] positionsy, double[] velocitiesx, double[] velocitiesy, double[] accelerationsx, 
            double[] accelerationsy, int width, int height, float mousex, float mousey, int[,] squareFish, int[] squaresStart,
            int[] fishInSquare, int squaresInRow, int squaresNumber, int[] bitmap)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            FishFunctions.AlignWithOtherFish(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, ind,
                squareFish, squaresStart, fishInSquare, squaresInRow, squaresNumber);
            FishFunctions.CohesionWithOtherFish(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, ind,
                squaresStart, squareFish, fishInSquare, squaresNumber, squaresInRow);
            FishFunctions.AvoidOtherFish(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, ind,
                squaresStart, squareFish, fishInSquare, squaresInRow, squaresNumber);
            if (mousex >= 0 && mousey >= 0)
                FishFunctions.AvoidMouse(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, ind,
                    mousex, mousey);
            FishFunctions.UpdateFish(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, ind,
                width, height, mousex, mousey);
            FishFunctions.Edges(positionsx, positionsy, ind, width, height);
            DeviceFunction.SyncThreads();
            int col = (255 << 24) + (0 << 16) + (255 << 8) + 255;
            int x = (int)positionsx[ind];
            int y = (int)positionsy[ind];
            CircleDrawing.CircleBresenham(x, y, 2, bitmap, width, height, col);
        }
    }
}
