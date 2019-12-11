using System;
using System.Drawing;
using System.Windows.Forms;
using System.Numerics;
using Alea.CSharp;
using Alea;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace FishShoal
{
    public partial class Form1 : Form
    {
        public static Vector2 Mouse { get; set; } = new Vector2(-1, -1);

        private bool gpu = true;
        private bool drawOnGpu = true;

        private const int fishNumber = 10240, squareSize = 100;
        private int windowWidth , windowHeight, squaresNumber, squaresInRow;
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

        public Form1()
        {
            InitializeComponent();
        }

        [GpuManaged]
        private void RunGpu()
        {
            var gpu = Alea.Gpu.Default;
            gpu.Launch(FishGPU.Kernel, new Alea.LaunchParam((int)Math.Ceiling((double)fishNumber / 512), 512), positionsx, positionsy,
                velocitiesx, velocitiesy, accelerationsx, accelerationsy, windowWidth, windowHeight, Mouse.X,
                Mouse.Y, squarefish, squarestart, fishinsquere, squaresInRow, squaresNumber, bitmap);
            FishFunctions.UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, fishNumber,
                squaresNumber, squareSize, windowWidth);
            if (!drawOnGpu)
                pictureBox1.Invalidate();
            else
                DrawFish();
        }

        private void RunCpu()
        {
            for (int i = 0; i < fishNumber; i++)
                FishCPU.KernelCpu(positionsx, positionsy, velocitiesx, velocitiesy, accelerationsx,
                    accelerationsy, windowWidth, windowHeight, Mouse.X, Mouse.Y, squarefish, squarestart,
                    fishinsquere, squaresInRow, squaresNumber, i, bitmap);
            FishFunctions.UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, fishNumber,
                squaresNumber, squareSize, windowWidth);
            pictureBox1.Invalidate();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            if (gpu)
                RunGpu();
            else
                RunCpu();
        }
        
        private void OnPictureBoxMouseMove(object sender, MouseEventArgs e)
        {
            Mouse = new Vector2(e.Location.X, e.Location.Y);
        }

        private void OnFormLoad(object sender, EventArgs e)
        {
            random = new Random();
            pictureBox1.SizeMode = PictureBoxSizeMode.Normal;
            windowWidth = pictureBox1.Width;
            windowHeight = pictureBox1.Height;
            bitmap = new int[windowWidth * windowHeight];
            squaresNumber = (int)Math.Ceiling((double)windowHeight / squareSize) * (int)Math.Ceiling((double)windowWidth / squareSize);
            squaresInRow = (windowWidth / squareSize);
            positionsx = new double[fishNumber];
            positionsy = new double[fishNumber];
            velocitiesx = new double[fishNumber];
            velocitiesy = new double[fishNumber];
            accelerationsx = new double[fishNumber];
            accelerationsy = new double[fishNumber];
            fishinsquere = new int[fishNumber];
            squarestart = new int[squaresNumber];
            squarefish = new int[fishNumber, 2];

            for (int i = 0; i < fishNumber; i++)
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
            FishFunctions.UpdateSquares(squarefish, squarestart, fishinsquere, positionsx, positionsy, fishNumber,
                squaresNumber, squareSize, windowWidth);
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

        private void DrawFish(Graphics graphics = null)
        {
            if (drawOnGpu)
            {
                if (bitmap != null)
                {
                    Bitmap bmp = new Bitmap(windowWidth, windowHeight);
                    var data = bmp.LockBits(new Rectangle(Point.Empty, bmp.Size), ImageLockMode.WriteOnly,
                        PixelFormat.Format32bppRgb);
                    Marshal.Copy(bitmap, 0, data.Scan0, bitmap.Length);
                    bmp.UnlockBits(data);
                    pictureBox1.Image = bmp;
                }
            }
            else
            {
                for (int i = 0; i < fishNumber; i++)
                {
                    double x = positionsx[i];
                    double y = positionsy[i];
                    graphics.DrawEllipse(new Pen(Brushes.Orange, 1), (float)positionsx[i], (float)positionsy[i], 5, 5);
                }
            }
            bitmap = new int[windowWidth * windowHeight];
            label1.Text = clock.CalcFrames();
        }
    }
}
