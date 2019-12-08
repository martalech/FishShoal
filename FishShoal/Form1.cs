using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Numerics;
using Alea.CSharp;
using Alea.Parallel;
using Alea;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace FishShoal
{
    public partial class Form1 : Form
    {
        private bool GPU = true;
        private bool drawOnGpu = true;

        private int FishNumber = 20480, SquareSize = 100, WindowWidth, WindowHeight, SquaresNumber, SquaresInRow;
        public static Vector2 Mouse { get; set; } = new Vector2(-1, -1);
        private Clock clock = new Clock();

        private double[] positionsx;
        private double[] positionsy;
        private double[] velocitiesx;
        private double[] velocitiesy;
        private double[] accelerationsx;
        private double[] accelerationsy;
        private int[,] squarefish;
        private int[] fishinsquere;
        private int[] squarestart;
        private int[] bitmap;

        private Random random;

        private static void UpdateSquares(int[,] sf, int[] ss, int[] fis, double[] px, double[] py, int fish_number,
            int squares_number, int square_size, int window_width)
        {
            int[] keys = new int[fish_number];
            int[] values = new int[fish_number];
            for (int i = 0; i < fish_number; i++)
            {
                fis[i] = SquareID(px[i], py[i], square_size, window_width);
                sf[i, 0] = fis[i];
                sf[i, 1] = i;
                keys[i] = sf[i, 0];
                values[i] = sf[i, 1];
            }
            for (int i = 0; i < squares_number; i++)
            {
                ss[i] = -1;
            }
            Array.Sort(keys, values);
            for (int i = 0; i < fish_number; i++)
            {
                sf[i, 0] = keys[i];
                sf[i, 1] = values[i];
            }
            int j = 0;
            ss[0] = 0;
            for (int i = 0; i < sf.GetLength(0); i++)
            {
                if (sf[i, 0] != j)
                {
                    j = sf[i, 0];
                    ss[j] = i;
                }
            }
        }

        public static int SquareID(double x, double y, int square_size, int window_width)
        {
            int newx = (int)x / square_size;
            int newy = (int)y / square_size;
            return (int)(newy * (double)(window_width / square_size) + newx);
        }
        public Form1()
        {
            InitializeComponent();
        }

        private static void Kernel(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int width, int height, float mx, float my,
            int[,] sf, int[] ss, int[] fis, int sir, int sn, int[] bitmap)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            AlignWithOtherFish(px, py, vx, vy, ax, ay, ind, sf, ss, fis, sir, sn);
            DeviceFunction.SyncThreads();
            CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind, ss, sf, fis, sn, sir);
            DeviceFunction.SyncThreads();
            AvoidhOtherFish(px, py, vx, vy, ax, ay, ind, ss, sf, fis, sir, sn);
            if (mx >= 0 && my >= 0)
                AvoidMouse(px, py, vx, vy, ax, ay, ind, mx, my);
            DeviceFunction.SyncThreads();
            UpdateFish(px, py, vx, vy, ax, ay, ind, width, height, mx, my);
            Edges(px, py, ind, width, height);
            DeviceFunction.SyncThreads();
            int col = (0 << 24) + (0 << 16) + (255 << 8) + 255;
            int x = (int)px[ind];
            int y = (int)py[ind];
            circleBres(x, y, 2, bitmap, width, height, col);
        }

        private static void KernelCpu(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int width, int height, float mx, float my,
            int[,] sf, int[] ss, int[] fis, int sir, int sn, int ind, int[] bitmap)
        {
            AlignWithOtherFish(px, py, vx, vy, ax, ay, ind, sf, ss, fis, sir, sn);
            CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind, ss, sf, fis, sn, sir);
            AvoidhOtherFish(px, py, vx, vy, ax, ay, ind, ss, sf, fis, sir, sn);
            if (mx >= 0 && my >= 0)
                AvoidMouse(px, py, vx, vy, ax, ay, ind, mx, my);
            UpdateFish(px, py, vx, vy, ax, ay, ind, width, height, mx, my);
            Edges(px, py, ind, width, height);
            int col = (0 << 24) + (0 << 16) + (255 << 8) + 255;
            int x = (int)px[ind];
            int y = (int)py[ind];
            circleBres(x, y, 2, bitmap, width, height, col);
        }

        [GpuManaged]
        private void RunGpu()
        {
            var gpu = Alea.Gpu.Default;
            gpu.Launch(Kernel, new Alea.LaunchParam(40, 512), positionsx, positionsy,
                velocitiesx, velocitiesy, accelerationsx, accelerationsy, WindowWidth, WindowHeight, Mouse.X, Mouse.Y, squarefish, squarestart,
                fishinsquere, SquaresInRow, SquaresNumber, bitmap);
            UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, FishNumber, SquaresNumber, SquareSize, WindowWidth);
            if (!drawOnGpu)
                pictureBox1.Invalidate();
            else
                DrawFish();
        }

        private void RunCpu()
        {
            for (int i = 0; i < FishNumber; i++)
            {
                KernelCpu(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx, accelerationsy, WindowWidth,
                    WindowHeight, Mouse.X, Mouse.Y, squarefish, squarestart, fishinsquere, SquaresInRow, SquaresNumber, i, bitmap);
            }
            UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, FishNumber, SquaresNumber, SquareSize, WindowWidth);
            pictureBox1.Invalidate();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            if (GPU)
                RunGpu();
            else
                RunCpu();
        }

        private static void AlignHelperFunc(int s_id, double[] px, double[] py, double[] vx,
            double[] vy, int ind, int[,] sf, int[] ss, int squares_number, ref int nc, ref double steeringx, ref double steeringy)
        {
            if (s_id < 0 || s_id >= squares_number)
                return;
            int i = ss[s_id];
            if (i >= 0)
            {
                while (sf[i, 0] == s_id)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 50)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            nc++;
                        }
                    }
                    i++;
                    if (i >= squares_number)
                        return;
                }
            }
        }

        private static bool IsInBorder(int x, int y, int width, int height)
        {
            return x > 0 && x < width && y > 0 && y < height;
        }

        private static void drawCircle(int xc, int yc, int x, int y, int width, int height, int col, int[] bitmap)
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

        private static void circleBres(int xc, int yc, int r, int[] bitmap, int width, int height, int col)
        {
            int x = 0, y = r;
            int d = 3 - 2 * r;
            drawCircle(xc, yc, x, y, width, height, col, bitmap);
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
                drawCircle(xc, yc, x, y, width, height, col, bitmap);
            }
        }

        private static void AlignWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind,
            int[,] sf, int[] ss, int[] fis, int squaresinrow, int squaresnumber)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareid = fis[ind];
            int lu = squareid - squaresinrow - 1, lm = squareid - squaresinrow, lr = squareid - squaresinrow + 1,
                ml = squareid - 1, mr = squareid + 1, dl = squareid + squaresinrow - 1, dm = squareid + squaresinrow, dr = squareid + squaresinrow + 1;

            AlignHelperFunc(squareid, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(lu, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(lm, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(lr, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(ml, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(mr, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(dr, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(dm, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);
            AlignHelperFunc(dl, px, py, vx, vy, ind, sf, ss, squaresnumber, ref neighboursCount, ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float maxValue = 4;
                double vLength = vectorlength(0, 0, steeringx, steeringy);
                if (vLength > 0)
                    steeringx *= (maxValue / vLength);
                if (vLength > 0)
                    steeringy *= (maxValue / vLength);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    maxValue = 0.2f;
                    if (vLength > 0)
                        steeringx *= (maxValue / vLength);
                    if (vLength > 0)
                        steeringy *= (maxValue / vLength);
                }
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static double vectorlength(double x1, double y1, double x2, double y2)
        {
            return DeviceFunction.Sqrt((DeviceFunction.Abs(x2 - x1) * DeviceFunction.Abs(x2 - x1))
                + (DeviceFunction.Abs(y2 - y1) * DeviceFunction.Abs(y2 - y1)));
        }

        private static void CohesionHelperFunc(int s_id, double[] px, double[] py, int ind, int[] ss, int [,] sf, int[] fis,
            int squaresnumber, ref int nc, ref double steeringx, ref double steeringy)
        {
            if (s_id < 0 || s_id >= squaresnumber)
                return;
            int i = ss[s_id];
            if (i >= 0)
            {
                while (i < squaresnumber && sf[i, 0] == s_id)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 100)
                        {
                            steeringx += px[sf[i, 1]];
                            steeringy += py[sf[i, 1]];
                            nc++;
                        }
                    }
                    i++;
                    if (i >= squaresnumber)
                        return;
                }
            }
        }

        private static void CohesionWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, int[] ss,
            int [,] sf, int[] fis, int sn, int sir)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareid = fis[ind];
            int lu = squareid - sir - 1, lm = squareid - sir, lr = squareid - sir + 1,
                ml = squareid - 1, mr = squareid + 1, dl = squareid + sir - 1, dm = squareid + sir, dr = squareid + sir + 1;

            CohesionHelperFunc(squareid, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(lu, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(lm, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(lr, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(ml, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(mr, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(dr, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(dm, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);
            CohesionHelperFunc(dl, px, py, ind, ss, sf, fis, sn, ref neighboursCount, ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                steeringx -= px[ind];
                steeringy -= py[ind];
                float maxValue = (float)4;
                double vLength = vectorlength(0, 0, steeringx, steeringy);
                if (vLength > 0)
                    steeringx *= (maxValue / vLength);
                if (vLength > 0)
                    steeringy *= (maxValue / vLength);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    vLength = vectorlength(0, 0, steeringx, steeringy);
                    maxValue = (float)0.2;
                    if (vLength > 0)
                        steeringx *= (maxValue / vLength);
                    if (vLength > 0)
                        steeringy *= (maxValue / vLength);
                }
            }

            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void AvoidHelperFunc(int s_id, double[] px, double[] py, int ind, int[] ss,
            int[,] sf, int squaresnumber, ref int nc, ref double steeringx, ref double steeringy)
        {
            if (s_id < 0 || s_id >= squaresnumber)
                return;
            int i = ss[s_id];
            if (i >= 0)
            {
                while (i < squaresnumber && sf[i, 0] == s_id)
                {
                    if (sf[i, 1] != ind)
                    {
                        double distance;
                        if ((distance = vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]])) <= 80)
                        {
                            double diffx = px[ind] - px[sf[i, 1]];
                            double diffy = py[ind] - py[sf[i, 1]];
                            if (distance > 0)
                                diffx /= distance;
                            if (distance > 0)
                                diffy /= distance;
                            steeringx += diffx;
                            steeringy += diffy;
                            nc++;
                        }
                    }
                    i++;
                    if (i >= squaresnumber)
                        return;
                }
            }
        }

        private static void AvoidhOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, int[] ss,
            int[,] sf, int[] fis, int sir, int sn)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareid = fis[ind];
            int lu = squareid - sir - 1, lm = squareid - sir, lr = squareid - sir + 1,
                ml = squareid - 1, mr = squareid + 1, dl = squareid + sir - 1, dm = squareid + sir, dr = squareid + sir + 1;

            AvoidHelperFunc(squareid, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(lu, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(lm, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(lr, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(dm, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(dr, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(dl, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(ml, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);
            AvoidHelperFunc(mr, px, py, ind, ss, sf, sn, ref neighboursCount, ref steeringx, ref steeringy);

            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float maxValue = (float)4;
                double vLength = vectorlength(0, 0, steeringx, steeringy);
                if (vLength > 0)
                    steeringx *= (maxValue / vLength);
                if (vLength > 0)
                    steeringy *= (maxValue / vLength);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    vLength = vectorlength(0, 0, steeringx, steeringy);
                    maxValue = (float)0.2;
                    if (vLength > 0)
                        steeringx *= (maxValue / vLength);
                    if (vLength > 0)
                        steeringy *= (maxValue / vLength);
                }
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void AvoidMouse(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, float mousex, float mousey)
        {
            double steeringx = 0, steeringy = 0;
            double distance = vectorlength(px[ind], py[ind], mousex, mousey);

            if (distance <= 40)
            {
                steeringx = px[ind] - mousex;
                steeringy = py[ind] - mousey;
                if (distance > 0)
                {
                    steeringx /= distance;
                    steeringy /= distance;
                }
                float maxValue = (float)4;
                double vLength = vectorlength(0, 0, steeringx, steeringy);
                if (vLength > 0)
                    steeringx *= (maxValue / vLength);
                if (vLength > 0)
                    steeringy *= (maxValue / vLength);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void Edges(double[] px, double[] py, int ind, int width, int height)
        {
            if (px[ind] > width)
            {
                px[ind] = 0;
            }
            else if (px[ind] < 0)
            {
                px[ind] = width;
            }
            if (py[ind] > height)
            {
                py[ind] = 0;
            }
            else if (py[ind] < 0)
            {
                py[ind] = height;
            }
        }

        private void OnPictureBoxMouseMove(object sender, MouseEventArgs e)
        {
            Mouse = new Vector2(e.Location.X, e.Location.Y);
        }

        private void DrawFish(Graphics graphics = null)
        {
            if (drawOnGpu)
            {
                if (bitmap != null)
                {
                    Bitmap bmp = new Bitmap(WindowWidth, WindowHeight);
                    var data = bmp.LockBits(new Rectangle(Point.Empty, bmp.Size), ImageLockMode.WriteOnly, PixelFormat.Format32bppRgb);
                    Marshal.Copy(bitmap, 0, data.Scan0, bitmap.Length);
                    bmp.UnlockBits(data);
                    pictureBox1.Image = bmp;
                }
            }
            else
            {
                for (int i = 0; i < FishNumber; i++)
                {
                    var x = positionsx[i];
                    var y = positionsy[i];
                    graphics.DrawEllipse(new Pen(Brushes.Orange, 1), (float)positionsx[i], (float)positionsy[i], 5, 5);
                }
            }
            bitmap = new int[WindowWidth * WindowHeight];
            label1.Text = clock.CalcFrames();
        }

        private static void UpdateFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, int width, int height, float mx, float my)
        {
            px[ind] += vx[ind];
            py[ind] += vy[ind];
            vx[ind] += ax[ind];
            vy[ind] += ay[ind];
            if (vectorlength(0, 0, vx[ind], vy[ind]) > 4)
            {
                float maxValue = (float)4;
                double vLength = vectorlength(0, 0, vx[ind], vy[ind]);
                if (vLength > 0)
                    vx[ind] *= (maxValue / vLength);
                if (vLength > 0)
                    vy[ind] *= (maxValue / vLength);
            }
            ax[ind] = 0;
            ay[ind] = 0;
        }

        private void OnFormLoad(object sender, EventArgs e)
        {
            random = new Random();
            pictureBox1.SizeMode = PictureBoxSizeMode.Normal;
            WindowWidth = pictureBox1.Width;
            WindowHeight = pictureBox1.Height;
            bitmap = new int[WindowWidth * WindowHeight];
            SquaresNumber = (int)Math.Ceiling((double)WindowHeight / SquareSize) * (int)Math.Ceiling((double)WindowWidth / SquareSize);
            SquaresInRow = (WindowWidth / SquareSize);
            positionsx = new double[FishNumber];
            positionsy = new double[FishNumber];
            velocitiesx = new double[FishNumber];
            velocitiesy = new double[FishNumber];
            accelerationsx = new double[FishNumber];
            accelerationsy = new double[FishNumber];
            fishinsquere = new int[FishNumber];
            squarestart = new int[SquaresNumber];
            squarefish = new int[FishNumber, 2];

            for (int i = 0; i < FishNumber; i++)
            {
                var r1 = random.Next(-10, 10);
                var r2 = random.Next(-10, 10);
                var Velocity = new Vector2(r1, r2);
                if (Velocity.Length() < 2 || Velocity.Length() > 4)
                {
                    float maxValue = (float)random.NextDouble() % 3 + 2;
                    float vLength = Velocity.Length();
                    if (vLength > 0)
                        Velocity *= new Vector2(maxValue / vLength, maxValue / vLength);
                }
                positionsx[i] = random.Next(1, pictureBox1.Width - 1);
                positionsy[i] = random.Next(1, pictureBox1.Height - 1);
                velocitiesx[i] = Velocity.X;
                velocitiesy[i] = Velocity.Y;
                accelerationsx[i] = 0;
                accelerationsy[i] = 0;
            }
            UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, FishNumber, SquaresNumber, SquareSize, WindowWidth);
            Timer timer = new Timer();
            timer.Interval = 10;
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private void OnPictureBoxMouseLeave(object sender, EventArgs e)
        {
            Mouse = new Vector2(-1, -1);
        }

        private void OnPictureBoxPaint(object sender, PaintEventArgs e)
        {
            if (!drawOnGpu)
            {
                var graphics = e.Graphics;
                DrawFish(graphics);
            }
        }
    }
}
