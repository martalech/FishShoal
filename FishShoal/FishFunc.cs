using Alea;
using System;

namespace FishShoal
{
    public static class FishFunctions
    {
        public static void UpdateSquares(int[,] squareFish, int[] squaresStart, int[] fishInSquare,
            double[] positionx, double[] positiony, int fish_number, int squaresNumber, int squareSize,
            int windowWidth)
        {
            int[] keys = new int[fish_number];
            int[] values = new int[fish_number];
            for (int i = 0; i < fish_number; i++)
            {
                fishInSquare[i] = SquareID(positionx[i], positiony[i], squareSize, windowWidth);
                squareFish[i, 0] = fishInSquare[i];
                squareFish[i, 1] = i;
                keys[i] = squareFish[i, 0];
                values[i] = squareFish[i, 1];
            }
            for (int i = 0; i < squaresNumber; i++)
                squaresStart[i] = -1;
            Array.Sort(keys, values);
            for (int i = 0; i < fish_number; i++)
            {
                squareFish[i, 0] = keys[i];
                squareFish[i, 1] = values[i];
            }
            int j = 0;
            squaresStart[0] = 0;
            for (int i = 0; i < squareFish.GetLength(0); i++)
            {
                if (squareFish[i, 0] != j)
                {
                    j = squareFish[i, 0];
                    squaresStart[j] = i;
                }
            }
        }

        public static void UpdateFish(double[] positionx, double[] positiony, double[] velocityx, double[] velocityy,
            double[] accelerationx, double[] accelerationy, int ind, int width, int height, float mx, float my)
        {
            positionx[ind] += velocityx[ind];
            positiony[ind] += velocityy[ind];
            velocityx[ind] += accelerationx[ind];
            velocityy[ind] += accelerationy[ind];
            if (VectorLength(0, 0, velocityx[ind], velocityy[ind]) > 4)
            {
                float maxValue = (float)4;
                double vectorLength = VectorLength(0, 0, velocityx[ind], velocityy[ind]);
                if (vectorLength > 0)
                    velocityx[ind] *= (maxValue / vectorLength);
                if (vectorLength > 0)
                    velocityy[ind] *= (maxValue / vectorLength);
            }
            accelerationx[ind] = 0;
            accelerationy[ind] = 0;
        }

        private static void AlignHelperFunc(int squareId, double[] positionx, double[] positiony, double[] velocityx,
            double[] velocityy, int ind, int[,] squareFish, int[] squaresStart, int squaresNumber, ref int neighboursCount,
            ref double steeringx, ref double steeringy)
        {
            if (squareId < 0 || squareId >= squaresNumber)
                return;
            int i = squaresStart[squareId];
            if (i >= 0)
            {
                while (squareFish[i, 0] == squareId)
                {
                    if (squareFish[i, 1] != ind)
                    {
                        if (VectorLength(positionx[ind], positiony[ind], positionx[squareFish[i, 1]],
                            positiony[squareFish[i, 1]]) <= 50)
                        {
                            steeringx += velocityx[squareFish[i, 1]];
                            steeringy += velocityy[squareFish[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                    if (i >= squaresNumber)
                        return;
                }
            }
        }

        public static void AlignWithOtherFish(double[] positionx, double[] positiony, double[] velocityx,
            double[] velocityy, double[] accelerationx, double[] accelerationy, int ind, int[,] squareFish,
            int[] squaresStart, int[] fishInSquare, int squaresInRow, int squaresNumber)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareId = fishInSquare[ind];
            int topLeft = squareId - squaresInRow - 1, topMiddle = squareId - squaresInRow, topRight = squareId - squaresInRow + 1,
                middleLeft = squareId - 1, middleRight = squareId + 1, bottomLeft = squareId + squaresInRow - 1,
                bottomMiddle = squareId + squaresInRow, bottomRight = squareId + squaresInRow + 1;

            AlignHelperFunc(squareId, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(topLeft, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(topMiddle, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(topRight, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(middleLeft, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(middleRight, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(bottomRight, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(bottomMiddle, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(bottomLeft, positionx, positiony, velocityx, velocityy, ind, squareFish, squaresStart, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float maxValue = 4;
                double vectorLength = VectorLength(0, 0, steeringx, steeringy);
                if (vectorLength > 0)
                    steeringx *= (maxValue / vectorLength);
                if (vectorLength > 0)
                    steeringy *= (maxValue / vectorLength);
                steeringx -= velocityx[ind];
                steeringy -= velocityy[ind];
                if (VectorLength(0, 0, steeringx, steeringy) > 0.2)
                {
                    maxValue = 0.2f;
                    if (vectorLength > 0)
                        steeringx *= (maxValue / vectorLength);
                    if (vectorLength > 0)
                        steeringy *= (maxValue / vectorLength);
                }
            }
            accelerationx[ind] += steeringx;
            accelerationy[ind] += steeringy;
        }

        private static void CohesionHelperFunc(int squareId, double[] positionx, double[] positiony, int ind,
            int[] squaresStart, int[,] squareFish, int squaresNumber, ref int neighboursCount,
            ref double steeringx, ref double steeringy)
        {
            if (squareId < 0 || squareId >= squaresNumber)
                return;
            int i = squaresStart[squareId];
            if (i >= 0)
            {
                while (i < squaresNumber && squareFish[i, 0] == squareId)
                {
                    if (squareFish[i, 1] != ind)
                    {
                        if (VectorLength(positionx[ind], positiony[ind], positionx[squareFish[i, 1]],
                            positiony[squareFish[i, 1]]) <= 100)
                        {
                            steeringx += positionx[i];
                            steeringy += positiony[i];
                            neighboursCount++;
                        }
                    }
                    i++;
                    if (i >= squaresNumber)
                        return;
                }
            }
        }

        public static void CohesionWithOtherFish(double[] positionx, double[] positiony, double[] velocityx,
            double[] velocityy, double[] accelerationx, double[] accelerationy, int ind, int[] squaresStart,
            int[,] squareFish, int[] fishInSquare, int squaresNumber, int squaresInRow)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareId = fishInSquare[ind];
            int topLeft = squareId - squaresInRow - 1, topMiddle = squareId - squaresInRow, topRight = squareId - squaresInRow + 1,
                middleLeft = squareId - 1, middleRight = squareId + 1, bottomLeft = squareId + squaresInRow - 1,
                bottomMiddle = squareId + squaresInRow, bottomRight = squareId + squaresInRow + 1;

            CohesionHelperFunc(squareId, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(topLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(topMiddle, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(topRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(middleLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(middleRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(bottomRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(bottomMiddle, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(bottomLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber,
                ref neighboursCount, ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                steeringx -= positionx[ind];
                steeringy -= positiony[ind];
                float maxValue = (float)4;
                double vectorLength = VectorLength(0, 0, steeringx, steeringy);
                if (vectorLength > 0)
                    steeringx *= (maxValue / vectorLength);
                if (vectorLength > 0)
                    steeringy *= (maxValue / vectorLength);
                steeringx -= velocityx[ind];
                steeringy -= velocityy[ind];
                if (VectorLength(0, 0, steeringx, steeringy) > 0.2)
                {
                    vectorLength = VectorLength(0, 0, steeringx, steeringy);
                    maxValue = (float)0.2;
                    if (vectorLength > 0)
                        steeringx *= (maxValue / vectorLength);
                    if (vectorLength > 0)
                        steeringy *= (maxValue / vectorLength);
                }
            }
            accelerationx[ind] += steeringx;
            accelerationy[ind] += steeringy;
        }

        private static void AvoidHelperFunc(int squareId, double[] positionx, double[] positiony, int ind, int[] squaresStart,
            int[,] squareFish, int squaresNumber, ref int neighboursCount, ref double steeringx, ref double steeringy)
        {
            if (squareId < 0 || squareId >= squaresNumber)
                return;
            int i = squaresStart[squareId];
            if (i >= 0)
            {
                while (i < squaresNumber && squareFish[i, 0] == squareId)
                {
                    if (squareFish[i, 1] != ind)
                    {
                        double distance;
                        if ((distance = VectorLength(positionx[ind], positiony[ind], positionx[squareFish[i, 1]],
                            positiony[squareFish[i, 1]])) <= 80)
                        {
                            double diffx = positionx[ind] - positionx[squareFish[i, 1]];
                            double diffy = positiony[ind] - positiony[squareFish[i, 1]];
                            if (distance > 0)
                                diffx /= distance;
                            if (distance > 0)
                                diffy /= distance;
                            steeringx += diffx;
                            steeringy += diffy;
                            neighboursCount++;
                        }
                    }
                    i++;
                    if (i >= squaresNumber)
                        return;
                }
            }
        }

        public static void AvoidOtherFish(double[] positionx, double[] positiony, double[] velocityx, double[] velocityy,
            double[] accelerationx, double[] accelerationy, int ind, int[] squaresStart, int[,] squareFish,
            int[] fishInSquare, int squaresInRow, int squaresNumber)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareId = fishInSquare[ind];
            int topLeft = squareId - squaresInRow - 1, topMiddle = squareId - squaresInRow, topRight = squareId - squaresInRow + 1,
                middleLeft = squareId - 1, middleRight = squareId + 1, bottomLeft = squareId + squaresInRow - 1,
                bottomMiddle = squareId + squaresInRow, bottomRight = squareId + squaresInRow + 1;

            AvoidHelperFunc(squareId, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(topLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(topMiddle, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(topRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(bottomMiddle, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(bottomRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(bottomLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(middleLeft, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);
            AvoidHelperFunc(middleRight, positionx, positiony, ind, squaresStart, squareFish, squaresNumber, ref neighboursCount,
                ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float maxValue = (float)4;
                double vectorLength = VectorLength(0, 0, steeringx, steeringy);
                if (vectorLength > 0)
                    steeringx *= (maxValue / vectorLength);
                if (vectorLength > 0)
                    steeringy *= (maxValue / vectorLength);
                steeringx -= velocityx[ind];
                steeringy -= velocityy[ind];
                if (VectorLength(0, 0, steeringx, steeringy) > 0.2)
                {
                    vectorLength = VectorLength(0, 0, steeringx, steeringy);
                    maxValue = (float)0.2;
                    if (vectorLength > 0)
                        steeringx *= (maxValue / vectorLength);
                    if (vectorLength > 0)
                        steeringy *= (maxValue / vectorLength);
                }
            }
            accelerationx[ind] += steeringx;
            accelerationy[ind] += steeringy;
        }

        public static void AvoidMouse(double[] positionx, double[] positiony, double[] velocityx, double[] velocityy,
            double[] accelerationx, double[] accelerationy, int ind, float mousex, float mousey)
        {
            double steeringx = 0, steeringy = 0;
            double distance = VectorLength(positionx[ind], positiony[ind], mousex, mousey);

            if (distance <= 40)
            {
                steeringx = positionx[ind] - mousex;
                steeringy = positiony[ind] - mousey;
                if (distance > 0)
                {
                    steeringx /= distance;
                    steeringy /= distance;
                }
                float maxValue = (float)4;
                double vectorLength = VectorLength(0, 0, steeringx, steeringy);
                if (vectorLength > 0)
                    steeringx *= (maxValue / vectorLength);
                if (vectorLength > 0)
                    steeringy *= (maxValue / vectorLength);
                steeringx -= velocityx[ind];
                steeringy -= velocityy[ind];
            }
            accelerationx[ind] += steeringx;
            accelerationy[ind] += steeringy;
        }

        public static void Edges(double[] positionx, double[] positiony, int ind, int width, int height)
        {
            if (positionx[ind] > width)
                positionx[ind] = 0;
            else if (positionx[ind] < 0)
                positionx[ind] = width;
            if (positiony[ind] > height)
                positiony[ind] = 0;
            else if (positiony[ind] < 0)
                positiony[ind] = height;
        }

        public static int SquareID(double x, double y, int squareSize, int windowWidth)
        {
            int newx = (int)x / squareSize;
            int newy = (int)y / squareSize;
            return (int)(newy * (double)(windowWidth / squareSize) + newx);
        }

        public static double VectorLength(double x1, double y1, double x2, double y2)
        {
            return DeviceFunction.Sqrt((DeviceFunction.Abs(x2 - x1) * DeviceFunction.Abs(x2 - x1))
                + (DeviceFunction.Abs(y2 - y1) * DeviceFunction.Abs(y2 - y1)));
        }
    }
}
