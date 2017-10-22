using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        
        static void Main(string[] args)
        {
            //JudgeSquareSum(0);
            //LogSystem ls = new LogSystem();
            //ls.Put(1, "2017:01:01:23:59:59");
            //ls.Put(2, "2017:01:02:23:59:59");
            //ls.Retrieve("2017:01:01:23:59:58", "2017:01:02:23:59:58", "Second");
            int[] tz = new int[] { 8860, -853, 6534, 4477, -4589, 8646, -6155, -5577, -1656, -5779, -2619, -8604, -1358, -8009, 4983, 7063, 3104, -1560, 4080, 2763, 5616, -2375, 2848, 1394, -7173, -5225, -8244, -809, 8025, -4072, -4391, -9579, 1407, 6700, 2421, -6685, 5481, -1732, -8892, -6645, 3077, 3287, -4149, 8701, -4393, -9070, -1777, 2237, -3253, -506, -4931, -7366, -8132, 5406, -6300, -275, -1908, 67, 3569, 1433, -7262, -437, 8303, 4498, -379, 3054, -6285, 4203, 6908, 4433, 3077, 2288, 9733, -8067, 3007, 9725, 9669, 1362, -2561, -4225, 5442, -9006, -429, 160, -9234, -4444, 3586, -5711, -9506, -79, -4418, -4348, -5891 };
            int[] x = new int[] {1, 12, -5, -6, 50, 3};
            FindMaxAverage1(x, 4);
            FindMaxAverage1(tz, 93);
            Console.ReadKey();
        }



        public static double FindMaxAverage1(int[] nums, int k)
        {
            
            int b = 0;
            for (int n = 0; n < k ; n++)
            {
                b += nums[n];
            }
            int sum = b;
            for (int i = 1; i < nums.Length; i++)
            {               
                if (i + k-1 < nums.Length)
                {                    
                    b = b - nums[i - 1] + nums[i + k - 1];
                }
                if (sum < b)
                    sum = b;
            }
            return (double)sum / k;
        }
        public static bool JudgeSquareSum(int c)
        {
            for (var i = 0; i <= c; i++)
            {
                var m = c - i * i;
                var n = Math.Sqrt(m);
                if (n < i) break;
                if ((int)n==n) return true;
            }
            return false;
        }

        //public static int[] SmallestRange(IList<IList<int>> nums)
        //{
        //    int[] result = new int[2];
        //    var minMax = 1000000;
        //    var maxMin = 0;
            
        //    foreach (var il in nums)
        //    {
        //        if (il[0] > maxMin) maxMin = il[0];
        //        if (il[il.Count] < minMax) minMax = il[il.Count];
        //    }
        //    result[0] = maxMin;
        //    result[1] = minMax;

        //}

        //public static int[] SmallestRange(IList<IList<int>> nums)
        //{
            
        //    List<int> sumlist = new List<int>();
        //    sumlist = nums.Aggregate(sumlist, (current, ls) => current.Concat(ls).ToList());
        //    Array.Sort(sumlist.ToArray());
        //    int count = sumlist.Count;
        //    int[] result = { sumlist[0],sumlist[count-1]};
        //    for (var i = 0; i < count; i++)
        //    {
                
        //    }
        //}

        public static bool Contains(IList<int> list, int min,int max)
        {
            return list.Any(i => i >= min && i <= max);
        }

        public static bool Smaller(int[] origin, int[] target)
        {
            if (origin.Length != 2 || target.Length != 2) return false;
            if (origin[1] - origin[0] < target[1] - target[0] ||
                ((origin[1] - origin[0] == target[1] - target[0]) && origin[0] < target[0]))
            {
                return true;
            }
            return false;
        }
    }


    /*["LogSystem","put","put","retrieve"]
    [[],[1,"2017:01:01:23:59:59"],[2,"2017:01:02:23:59:59"],["2017:01:01:23:59:58","2017:01:02:23:59:58","Second"]]*/
    public class LogSystem
    {
        private Dictionary<int, string> _logs;
        public LogSystem()
        {
            _logs = new Dictionary<int, string>();
        }

        public void Put(int id, string timestamp)
        {
            if (_logs.ContainsKey(id) && String.IsNullOrEmpty(timestamp)) return;
          
                _logs.Add(id, timestamp);
        }

        public IList<int> Retrieve(string s, string e, string gra)
        {
            var result = new List<int>();
            var start = s.Split(':');
            var end = e.Split(':');
            foreach (var item in _logs)
            {               
                int index;
                if (string.IsNullOrEmpty(gra)) index = 5;
                else index = (int)Enum.Parse(typeof(Gra), gra);
                var o = item.Value.Split(':');
                var r = Compare(o, start, index) && Compare(end, o, index);
                if (r) result.Add(item.Key);
            }
            return result;
        }

        public bool Compare(string[] origin, string[] target,int index)
        {
            bool result = true;
            for (var i = 0; i <= index; i++)
            {
                var o = Convert.ToInt32(origin[i]);
                var t = Convert.ToInt32(target[i]);
                if (o > t)
                {
                    break;
                }
                else if (o < t)
                {
                    result = false;
                    break;
                }
            }
            return result;
        }

        public enum Gra
        {
            Year,
            Month,
            Day,
            Hour,
            Minute,
            Second
        }
    }
}
