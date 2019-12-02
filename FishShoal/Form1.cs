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

namespace FishShoal
{
    public partial class Form1 : Form
    {
        public static Vector2 Mouse { get; set; } = new Vector2(-1, -1);
        //public static List<Fish> Fishes { get; set; } = new List<Fish>();
        public double[] positionsx = new double[1024];
        public double[] positionsy = new double[1024];
        public double[] velocitiesx = new double[1024];
        public double[] velocitiesy = new double[1024];
        public double[] accelerationsx = new double[1024];
        public double[] accelerationsy = new double[1024];
        public int[,] squarefish = new int[1024, 2];
        public int[] fishinsquere = new int[1024];
        public int[,] squarestart = new int[1501, 2];
        public Random random;
        public static int width, height;
        //private const int Length = 150;
        private static void UpdateSquares(int[,] sf, int[,] ss)
        {
            int[] keys = new int[1024];
            int[] values = new int[1024];
            for (int i = 0; i < 1024; i++)
            {
                keys[i] = sf[i, 0];
                values[i] = sf[i, 1];
            }
            for (int i = 0; i < 1501; i++)
            {
                ss[i, 0] = i;
                ss[i, 1] = -1;
            }
            Array.Sort(keys, values);
            for (int i = 0; i < 1024; i++)
            {
                sf[i, 0] = keys[i];
                sf[i, 1] = values[i];
            }
            int j = 0;
            ss[0, 0] = 0;
            ss[0, 1] = 0;
            for (int i = 0; i < sf.GetLength(0); i++)
            {
                if (sf[i, 0] != j)
                {
                    j = sf[i, 0];
                    ss[j, 0] = j;
                    ss[j, 1] = i;
                }
            }
        }

        public static int SquareID(double x, double y)
        {
            int newx = (int)x / 30;
            int newy = (int)y / 30;
            return newy * 50 + newx;
        }
        public Form1()
        {
            InitializeComponent();
            //random = new Random();
            //Fishes = new List<Fish>();
            //width = pictureBox1.Width;
            //height = pictureBox1.Height;
            //for (int i = 0; i < 1024; i++)
            //{
            //    var r1 = random.Next(-10, 10);
            //    var r2 = random.Next(-10, 10);
            //    var Velocity = new Vector2(r1, r2);
            //    if (Velocity.Length() < 2 || Velocity.Length() > 4)
            //    {
            //        float maxValue = (float)random.NextDouble() % 3 + 2;
            //        float vLength = Velocity.Length();
            //        if (vLength > 0)
            //            Velocity *= new Vector2(maxValue / vLength, maxValue / vLength);
            //    }
            //    positionsx[i] = random.Next(1, pictureBox1.Width - 1);
            //    positionsy[i] = random.Next(1, pictureBox1.Height - 1);
            //    squarefish[i, 0] = SquareID(positionsx[i], positionsy[i]);
            //    fishinsquere[i] = SquareID(positionsx[i], positionsy[i]);
            //    squarefish[i, 1] = i;
            //    velocitiesx[i] = Velocity.X;
            //    velocitiesy[i] = Velocity.Y;
            //    accelerationsx[i] = 0;
            //    accelerationsy[i] = 0;
            //}
            //UpdateSquares(squarefish, squarestart);

            //Timer timer = new Timer();
            //timer.Interval = 10;
            //timer.Tick += Timer_Tick;
            //timer.Start();
        }

        private static void Kernel(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int width, int height, float mx, float my,
            int[,] sf, int[,] ss, int[] fis)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            Edges(px, py, ind, width, height);
            DeviceFunction.SyncThreads();
            AlignWithOtherFish(px, py, vx, vy, ax, ay, ind, sf, ss, fis);
            DeviceFunction.SyncThreads();
            CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind);
            DeviceFunction.SyncThreads();
            AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            if (mx >= 0 && my >= 0)
                AvoidMouse(px, py, vx, vy, ax, ay, ind, mx, my);
            //Edges(px, py, ind, width, height);
            //DeviceFunction.SyncThreads();
            //Flock(px, py, vx, vy, ax, ay, ind, mx, my);
            DeviceFunction.SyncThreads();
            UpdateFish(px, py, vx, vy, ax, ay, ind, width, height, mx, my);
            //Alea.Gpu.Default.Synchronize();
            //pictureBox.Invalidate();
        }

        [GpuManaged]
        private void RunGpu()
        {
            var gpu = Alea.Gpu.Default;
            gpu.Launch(Kernel, new Alea.LaunchParam(2, 512), positionsx, positionsy,
                velocitiesx, velocitiesy, accelerationsx, accelerationsy, width, height, Mouse.X, Mouse.Y, squarefish, squarestart,
                fishinsquere);
            UpdateSquares(squarefish, squarestart);
            pictureBox1.Invalidate();
        }
        private void Timer_Tick(object sender, EventArgs e)
        {
            RunGpu();
        }

        private static void AlignWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind,
            int[,] sf, int[,] ss, int[] fis)
        {
            //double[] steering = { 0, 0 };
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            int squareid = fis[ind];
            if (ind == 20)
                Console.WriteLine("square: {0}", squareid);
            int lu = squareid - 50 - 1, lm = squareid - 50, lr = squareid - 50 + 1,
                ml = squareid - 1, mr = squareid + 1, dl = squareid + 50 - 1, dm = squareid + 50, dr = squareid + 50 + 1;
            int s1 = -1, s2 = -1, s3 = -1, s4 = -1, s5 = -1, s6 = -1, s7 = -1, s8 = -1, s9 = - 1;
            if (lu >= 0 && lu < 1500)
            {
                s1 = ss[lu, 1];
                if (ind == 20)
                    Console.WriteLine("startlu: {0}", s1);
            }
            if (lm >= 0 && lm < 1500)
            {
                s2 = ss[lm, 1];
                if (ind == 20)
                    Console.WriteLine("startum: {0}", s2);
            }
            if (lr >= 0 && lr < 1500)
            {
                s3 = ss[lr, 1];
                if (ind == 20)
                    Console.WriteLine("startur: {0}", s3);
            }
            if (ml >= 0 && ml < 1500)
            {
                s4 = ss[ml, 1];
                if (ind == 20)
                    Console.WriteLine("startml: {0}", s4);
            }
            if (squareid > 0 && squareid < 1500)
            {
                s5 = ss[squareid, 1];
                if (ind == 20)
                    Console.WriteLine("start: {0}", s5);
            }
            if (mr >= 0 && mr < 1500)
            {
                s6 = ss[mr, 1];
                if (ind == 20)
                    Console.WriteLine("startmr: {0}", s6);
            }
            if (dl >= 0 && dl < 1500)
            {
                s7 = ss[dl, 1];
                if (ind == 20)
                    Console.WriteLine("startdl: {0}", s7);
            }
            if (dm > 0 && dm < 1501)
            {
                s8 = ss[dm, 1];
                if (ind == 20)
                    Console.WriteLine("startdm: {0}", s8);
            }
            if (dr >= 0 && dr < 1500)
            {
                s9 = ss[dr, 1];
                if (ind == 20)
                    Console.WriteLine("startdr: {0}", s9);
            }

            int i = s1;
            if (i >= 0)
            {
                while (sf[i, 0] == lu)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins1: {0}", i - s1);
            i = s2;
            if (i >= 0)
            {
                while (sf[i, 0] == lm)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins2: {0}", i - s2);
            i = s3;
            if (i >= 0)
            {
                while (sf[i, 0] == lr)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins3: {0}", i - s3);

            i = s4;
            if (i >= 0)
            {
                while (sf[i, 0] == ml)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins4: {0}", i - s4);

            i = s5;
            if (i >= 0)
            {
                while (sf[i, 0] == squareid)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins5: {0}", i - s5);

            i = s6;
            if (i >= 0)
            {
                while (sf[i, 0] == mr)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins6: {0}", i - s6);

            i = s7;
            if (i >= 0)
            {
                while (sf[i, 0] == dl)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins7: {0}", i - s7);

            i = s8;
            if (i >= 0)
            {
                while (sf[i, 0] == dm)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins8: {0}", i - s8);

            i = s9;
            if (i >= 0)
            {
                while (sf[i, 0] == dr)
                {
                    if (sf[i, 1] != ind)
                    {
                        if (vectorlength(px[ind], py[ind], px[sf[i, 1]], py[sf[i, 1]]) <= 30)
                        {
                            steeringx += vx[sf[i, 1]];
                            steeringy += vy[sf[i, 1]];
                            neighboursCount++;
                        }
                    }
                    i++;
                }
            }
            if (ind == 20)
                Console.WriteLine("fishins9: {0}", i - s9);

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
            //
        }

        private static double vectorlength(double x1, double y1, double x2, double y2)
        {
            var lol = DeviceFunction.Sqrt((DeviceFunction.Abs(x2 - x1) * DeviceFunction.Abs(x2 - x1))
                + (DeviceFunction.Abs(y2 - y1) * DeviceFunction.Abs(y2 - y1)));
           return lol;
        }

        private static void CohesionWithOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            for (int i = 0; i < 1024; i++)
            {
                if (i != ind)
                {
                    if (vectorlength(px[ind], py[ind], px[i], py[i]) <= 30)
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
                steeringx -= px[ind];
                steeringy -= py[ind];
                float maxValue = (float)4;
                double vLength = vectorlength(0, 0, steeringx, steeringy);
                if (vLength > 0)
                    steeringx *= (maxValue / vLength);
                if (vLength > 0)
                    steeringy *= (maxValue / vLength);
                //steering *= new Vector2(maxValue / vLength, maxValue / vLength);
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

        private static void AvoidhOtherFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind)
        {
            double steeringx = 0, steeringy = 0;
            int neighboursCount = 0;
            for (int i = 0; i < 1024; i++)
            {
                if (ind != i)
                {
                    double distance;
                    if ((distance = vectorlength(px[ind], py[ind], px[i], py[i])) <= 30)
                    {
                        double diffx = px[ind] - px[i];
                        double diffy = py[ind] - py[i];
                        if (distance > 0)
                            diffx /= distance;
                        if (distance > 0)
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
                //if (steering.Length() > fish.MaxForce)
                //{
                //    vLength = steering.Length();
                //    maxValue = (float)fish.MaxForce;
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

        private static void Flock(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, float mx, float my,
            int[,] sf, int[,] ss, int[] fis)
        {
            AlignWithOtherFish(px, py, vx, vy, ax, ay, ind, sf, ss, fis);
            CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind);
            AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            //Vector2 fa4 = AvoidMouse(px, py, vx, vy, ax, ay, ind);
            //ax[ind] += (fa1.X + fa2.X + fa3.X + fa4.X);
            //ay[ind] += (fa1.Y + fa2.Y + fa3.Y + fa4.Y);
            //ax[ind] += fa1[0];
            //ay[ind] += fa1[1];
            //fish.Acceleration += AlignWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //fish.Acceleration += CohesionWithOtherFish(px, py, vx, vy, ax, ay, ind);
            //fish.Acceleration += AvoidhOtherFish(px, py, vx, vy, ax, ay, ind);
            if (mx >= 0 && my >= 0)
                AvoidMouse(px, py, vx, vy, ax, ay, ind, mx, my);
        }

        private void OnPictureBoxMouseMove(object sender, MouseEventArgs e)
        {
            Mouse = new Vector2(e.Location.X, e.Location.Y);
        }

        private void DrawFish(Graphics graphics)
        {
            for (int i = 0; i < 1024; i++)
            {
                var x = positionsx[i];
                var y = positionsy[i];
                graphics.DrawEllipse(new Pen(Brushes.Orange, 1), (float)positionsx[i], (float)positionsy[i], 5, 5);
            }
        }

        private static void UpdateFish(double[] px, double[] py, double[] vx, double[] vy, double[] ax, double[] ay, int ind, int width, int height, float mx, float my)
        {
            //Edges(px, py, ind, width, height);
            //Flock(px, py, vx, vy, ax, ay, ind, mx, my);
            px[ind] += vx[ind];
            py[ind] += vy[ind];
            //fish.Position += fish.Velocity;
            vx[ind] += ax[ind];
            vy[ind] += ay[ind];
            //fish.Velocity += fish.Acceleration;
            //double[] vel = { vx[ind], vy[ind] };
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
            random = new Random();
            width = pictureBox1.Width;
            height = pictureBox1.Height;
            for (int i = 0; i < 1024; i++)
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
                squarefish[i, 0] = SquareID(positionsx[i], positionsy[i]);
                fishinsquere[i] = SquareID(positionsx[i], positionsy[i]);
                squarefish[i, 1] = i;
                velocitiesx[i] = Velocity.X;
                velocitiesy[i] = Velocity.Y;
                accelerationsx[i] = 0;
                accelerationsy[i] = 0;
            }
            UpdateSquares(squarefish, squarestart);

            Timer timer = new Timer();
            timer.Interval = 10;
            timer.Tick += Timer_Tick;
            timer.Start();
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
