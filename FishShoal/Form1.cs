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

namespace FishShoal
{

    public partial class Form1 : Form
    {
        public static Vector2 Mouse { get; set; } = new Vector2(-1, -1);
        //public static List<Fish> Fishes { get; set; } = new List<Fish>();
        public double[] positionsx = new double[150];
        public double[] positionsy = new double[150];
        public double[] velocitiesx = new double[150];
        public double[] velocitiesy = new double[150];
        public double[] accelerationsx = new double[150];
        public double[] accelerationsy = new double[150];
        public Random random;
        public static int width, height;
        private const int Length = 150;
        public Form1()
        {
            InitializeComponent();
            random = new Random();
            //Fishes = new List<Fish>();
            for (int i = 0; i < 150; i++)
            {
                var r1 = random.Next(-100, 100);
                var r2 = random.Next(-100, 100);
                var Velocity = new Vector2(r1, r2);
                if (Velocity.Length() < 2 || Velocity.Length() > 4)
                {
                    float pozdro2 = (float)random.NextDouble() % 3 + 2;
                    float pozdro1 = Velocity.Length();
                    Velocity *= new Vector2(pozdro2 / pozdro1, pozdro2 / pozdro1);
                }
                positionsx[i] = random.Next(1, pictureBox1.Width - 1);
                positionsy[i] = random.Next(1, pictureBox1.Height - 1);
                velocitiesx[i] = Velocity.X;
                velocitiesy[i] = Velocity.Y;
                accelerationsx[i] = 0;
                accelerationsy[i] = 0;
            }
            //while (true)
            //{
            //    gpu.For();
            //}
            width = pictureBox1.Width;
            height = pictureBox1.Height;
            Timer timer = new Timer();
            timer.Interval = 20;
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private static void Kernel(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int width, int height, float mx, float my)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            Edges(px, py, ind, width, height);
            //DeviceFunction.SyncThreads();
            AlignWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //DeviceFunction.SyncThreads();
            AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            if (mx >= 0 && my >=0)
                AvoidMouse(px, py, vx, vy, ax, ay, ind, mx, my);
            UpdateFish(px, py, vx, vy, ax, ay, ind, width, height);
            if (ind == 50)
                Console.WriteLine("vx {0}, vy {1}", vx[ind], vy[ind]);
            //Alea.Gpu.Default.Synchronize();
            //pictureBox.Invalidate();
        }

        [GpuManaged]
        private void RunGpu()
        {
            var gpu = Alea.Gpu.Default;
            gpu.Launch(Kernel, new Alea.LaunchParam(1, 150), positionsx, positionsy,
                velocitiesx, velocitiesy, accelerationsx, accelerationsy, width, height, Mouse.X, Mouse.Y);
            pictureBox1.Invalidate();
        }
        private void Timer_Tick(object sender, EventArgs e)
        {
            RunGpu();
        }

        private static void AlignWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            //double[] steering = {0, 0 };
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            for (int i = 0; i < 150; i++)
            {
                if (i != ind)
                {
                    if (vectorlength(px[ind], py[ind], px[i], py[i]) <= 40)
                    {
                        steeringx += vx[i];
                        steeringy += vy[i];
                        neighboursCount++;
                    }
                }
            }
            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float pozdro2 = 4;
                double pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                steeringx *= (pozdro2 / pozdro1);
                steeringy *= (pozdro2 / pozdro1);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                //steering *= new Vector2(pozdro2 / pozdro1, pozdro2 / pozdro1);
                //steering -= new Vector2((float)vx[ind], (float)vy[ind]);
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                    pozdro2 = 0.2f;
                    steeringx *= (pozdro2 / pozdro1);
                    steeringy *= (pozdro2 / pozdro1);
                }
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
            //
        }

        private static double vectorlength(double x1, double y1, double x2, double y2)
        {

            return DeviceFunction.Sqrt((DeviceFunction.Abs(x2 - x1) * DeviceFunction.Abs(x2 - x1))
                + (DeviceFunction.Abs(y2 - y1) * DeviceFunction.Abs(y2 - y1)));
        }

        private static void CohesionWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            for (int i = 0; i < 150; i++)
            {
                if (i != ind)
                {
                    if (vectorlength(px[ind], py[ind], px[i], py[i]) <= 50)
                    {
                        steeringx += px[i];
                        steeringy += py[i];
                        neighboursCount++;
                    }
                }
            }
            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float pozdro2 = (float)4;
                double pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                steeringx -= px[ind];
                steeringy -= py[ind];
                steeringx *= (pozdro2 / pozdro1);
                steeringy *= (pozdro2 / pozdro1);
                //steering *= new Vector2(pozdro2 / pozdro1, pozdro2 / pozdro1);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                    pozdro2 = (float)0.2;
                    steeringx *= (pozdro2 / pozdro1);
                    steeringy *= (pozdro2 / pozdro1);
                }
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void AvoidhOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            for (int i = 0; i < 150; i++)
            {
                if (ind != i)
                {
                    double distance;
                    if ((distance = vectorlength(px[ind], py[ind], px[i], py[i])) <= 50)
                    {
                        double diffx = px[ind] - px[i];
                        double diffy = py[ind] - py[i];
                        diffx /= distance;
                        diffy /= distance;
                        steeringx += diffx;
                        steeringy += diffy;
                        neighboursCount++;
                    }
                }
            }
            if (neighboursCount > 0)
            {
                steeringx /= neighboursCount;
                steeringy /= neighboursCount;
                float pozdro2 = (float)4;
                double pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                steeringx *= (pozdro2 / pozdro1);
                steeringy *= (pozdro2 / pozdro1);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                if (vectorlength(0, 0, steeringx, steeringy) > 0.2)
                {
                    pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                    pozdro2 = (float)0.2;
                    steeringx *= (pozdro2 / pozdro1);
                    steeringy *= (pozdro2 / pozdro1);
                }
            }
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void AvoidMouse(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, float mousex, float mousey)
        {
            double steeringx = 0, steeringy = 0;
            //int neighboursCount = 0;
            //foreach (var otherFish in Fishes)
            //{
            //    if (fish != otherFish)
            //    {
            //        float distance;
            //        if ((distance = Vector2.Distance(fish.Position, otherFish.Position)) <= fish.NeighbourDistance)
            //        {
            //            Vector2 diff = new Vector2(fish.Position.X - otherFish.Position.X, fish.Position.Y - otherFish.Position.Y);
            //            diff /= distance;
            //            steering += diff;
            //            neighboursCount++;
            //        }
            //    }
            //}
            double distance = vectorlength(px[ind], py[ind], mousex, mousey);
            if (distance <= 30)
            {
                steeringx = px[ind] - mousex;
                steeringy = py[ind] - mousey;
                if (distance > 0)
                {
                    steeringx /= distance;
                    steeringy /= distance;
                }
                float pozdro2 = (float)4;
                double pozdro1 = vectorlength(0, 0, steeringx, steeringy);
                steeringx *= (pozdro2 / pozdro1);
                steeringy *= (pozdro2 / pozdro1);
                steeringx -= vx[ind];
                steeringy -= vy[ind];
                //if (steering.Length() > fish.MaxForce)
                //{
                //    pozdro1 = steering.Length();
                //    pozdro2 = (float)fish.MaxForce;
                //    steering *= new Vector2((float)fish.MaxForce / steering.Length(), (float)fish.MaxForce / steering.Length());
                //}
            }

            //if (neighboursCount > 0)
            //{
            //steering /= neighboursCount;

            //}
            ax[ind] += steeringx;
            ay[ind] += steeringy;
        }

        private static void Edges(double[] px, double[] py, int ind, int width, int height)
        {
            if (px[ind] > width)
            {
                px[ind] = 0;
            }
                //fish.Position = new Vector2(0, fish.Position.Y);
            else if (px[ind] < 0)
            {
                px[ind] = width;
            }
                //fish.Position = new Vector2(width, fish.Position.Y);
            if (py[ind] > height)
            {
                py[ind] = 0;
            }
                //fish.Position = new Vector2(fish.Position.X, 0);
            else if (py[ind] < 0)
            {
                py[ind] = height;
            }
                //fish.Position = new Vector2(fish.Position.X, height);
        }

        private static void Flock(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            //AlignWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            //Vector2 fa4 = AvoidMouse(px, py, vx, vy, ax, ay, ind);
            //ax[ind] += (fa1.X + fa2.X + fa3.X + fa4.X);
            //ay[ind] += (fa1.Y + fa2.Y + fa3.Y + fa4.Y);
            //ax[ind] += fa1[0];
            //ay[ind] += fa1[1];
            //fish.Acceleration += AlignWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //fish.Acceleration += CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //fish.Acceleration += AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            //fish.Acceleration += AvoidMouse(px, py, vx, vy, ax, ay, ind);
        }

        private void OnPictureBoxMouseMove(object sender, MouseEventArgs e)
        {
            Mouse = new Vector2(e.Location.X, e.Location.Y);
        }

        private void DrawFish(Graphics graphics)
        {
            for (int i = 0; i < 150; i++)
            {
                graphics.DrawEllipse(new Pen(Brushes.Orange, 1), (float)positionsx[i], (float)positionsy[i], 5, 5);
            }
        }

        private static void UpdateFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, int width, int height)
        {
            //Edges(px, py, ind, width, height);
            //Flock(px, py, vx, vy, ax, ay, ind);
            px[ind] += vx[ind];
            py[ind] += vy[ind];
            //fish.Position += fish.Velocity;
            vx[ind] += ax[ind];
            vy[ind] += ay[ind];
            //fish.Velocity += fish.Acceleration;
            //double[] vel = { vx[ind], vy[ind] };
            if (vectorlength(0, 0, vx[ind], vy[ind]) > 4)
            {
                float pozdro2 = (float)4;
                double pozdro1 = vectorlength(0, 0, vx[ind], vy[ind]);
                vx[ind] *= (pozdro2 / pozdro1);
                vy[ind] *= (pozdro2 / pozdro1);
            }
            ax[ind] = 0;
            ay[ind] = 0;
            //fish.Acceleration = new Vector2();
            //foreach(var fish in Fishes)
            //{
            //    List<Fish> neighbours = new List<Fish>();
            //    foreach(var otherFish in Fishes)
            //    {
            //        if (fish != otherFish)
            //        {

            //        }
            //    }
            //}
        }

        private void pictureBox1_LoadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            //RunGpu();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //RunGpu();
        }

        private void pictureBox1_MouseLeave(object sender, EventArgs e)
        {
            Mouse = new Vector2(-1, -1);
        }

        private void OnPictureBoxPaint(object sender, PaintEventArgs e)
        {
            var graphics = e.Graphics;
            DrawFish(graphics);
        }
    }
}
