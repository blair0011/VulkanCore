using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VulkanMinimalCompute
{
    class Program
    {
        static void Main(string[] args)
        {
            ComputeApplication app = new ComputeApplication();

            try
            {
                app.Run();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
            Console.WriteLine("Done.");
            Console.ReadKey();
        }       
    }
}
