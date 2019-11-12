using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace FishShoal
{
    //public class Fish
    //{
    //    public Vector2 Position { get; set; }
    //    public Vector2 Velocity { get; set; }
    //    public Vector2 Acceleration { get; set; } = new Vector2();
    //    public double Speed { get; set; }
    //    public int NeighbourDistance { get; set; }
    //    public double MaxForce { get; set; }
    //    public double MaxSpeed { get; set; }
    //    //public Vector2 AvarageHeading { get; set; }
    //    //public Vector2 AvaragePosition { get; set; }
        
    //    public Fish(int x, int y, Random random)
    //    {
    //        Position = new Vector2(x, y);
    //        MaxForce = 0.2;
    //        MaxSpeed = 4;
    //        var r1 = random.Next(-100, 100);
    //        var r2 = random.Next(-100, 100);
    //        Velocity = new Vector2(r1, r2);
    //        if (Velocity.Length() < 2 || Velocity.Length() > 4)
    //        {
    //            float pozdro2 = (float)random.NextDouble() % 3 + 2;
    //            float pozdro1 = Velocity.Length();
    //            Velocity *= new Vector2(pozdro2 / pozdro1, pozdro2 / pozdro1);
    //        }
    //        Speed = (double)random.Next(50, 100) / 100;
    //        NeighbourDistance = 40;
    //    }
    //}
}
