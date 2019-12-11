using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FishShoal
{
    public static class CircleDrawing
    {
        public static void CircleBresenham(int xc, int yc, int r, int[] bitmap, int width, int height, int col)
        {
            int x = 0, y = r;
            int d = 3 - 2 * r;
            DrawCircle(xc, yc, x, y, width, height, col, bitmap);
            while (y >= x)
            {
                x++;
                if (d > 0)
                {
                    y--;
                    d = d + 4 * (x - y) + 10;
                }
                else
                    d = d + 4 * x + 6;
                DrawCircle(xc, yc, x, y, width, height, col, bitmap);
            }
        }

        private static bool IsInBorder(int x, int y, int width, int height)
        {
            return x > 0 && x < width && y > 0 && y < height;
        }

        private static void DrawCircle(int xc, int yc, int x, int y, int width, int height, int col, int[] bitmap)
        {
            if (IsInBorder(xc + x, yc + y, width, height))
                bitmap[(yc + y) * width + xc + x] = col;
            if (IsInBorder(xc - x, yc + y, width, height))
                bitmap[(yc + y) * width + xc - x] = col;
            if (IsInBorder(xc + x, yc - y, width, height))
                bitmap[(yc - y) * width + xc + x] = col;
            if (IsInBorder(xc - x, yc - y, width, height))
                bitmap[(yc - y) * width + xc - x] = col;
            if (IsInBorder(xc + y, yc + x, width, height))
                bitmap[(yc + x) * width + xc + y] = col;
            if (IsInBorder(xc - y, yc + x, width, height))
                bitmap[(yc + x) * width + xc - y] = col;
            if (IsInBorder(xc + y, yc - x, width, height))
                bitmap[(yc - x) * width + xc + y] = col;
            if (IsInBorder(xc - y, yc - x, width, height))
                bitmap[(yc - x) * width + xc - y] = col;
        }
    }
}
