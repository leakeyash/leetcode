using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace algorithm
{
    public class CitiSolutions
    {
        public void Test()
        {
            var a = new int[] { 4, 5, 6, 7 };
            var b = new int[] { 1, 2, 3 };
            solution(a,b);
        }

        public int solution(int[] A)
        {
            var list = A.ToList();
            var count = 0;
            for(var i = 0; i < list.Count; i++)
            {
                var temp = list[i];
                list.RemoveAt(i);
                if (order(list))
                {
                    count++;
                }
                list.Insert(i, temp);
            }
            return count;
        }

        public bool order(List<int> list)
        {
            for(var i = 0; i < list.Count-1; i++)
            {
                if (list[i + 1] < list[i])
                {
                    return false;
                }
            }
            return true;
        }

        public int solution(int[] A, int[] B)
        {
            int n = A.Length;
            int m = B.Length;
            Array.Sort(A);
            Array.Sort(B);
            int i = 0;
            for (int k = 0; k < n;)
            {
                if (i < m - 1 && B[i] < A[k])
                    i += 1;
                else k++;
                if (A[k] == B[i])
                    return A[k];
            }
            return -1;
        }
    }
}
