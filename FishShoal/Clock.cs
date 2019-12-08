using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FishShoal
{
    class Clock
    {
        private DateTime lastTime;
        private int framesRendered;
        private int fps;

        public string CalcFrames()
        {
            framesRendered++;
            if ((DateTime.Now - lastTime).TotalSeconds >= 1)
            {
                fps = framesRendered;
                framesRendered = 0;
                lastTime = DateTime.Now;
            }
            return fps.ToString() + " FPS";
        }
    }
}
