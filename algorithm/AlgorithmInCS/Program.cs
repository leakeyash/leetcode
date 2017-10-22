using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace AlgorithmInCS
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //  //  [2147483647,2147483647,2147483647]
            //TreeNode root = new TreeNode(2147483647);
            //root.left = new TreeNode(2147483647);
            //root.right = new TreeNode(2147483647);
            Solution s = new Solution();
            //s.AverageOfLevels(root);
            //s.SolveEquation("3x=33+22+11");
            //s.FindErrorNums(new[] {2, 2});
            //s.FindLongestChain(new int[,] { { 3, 4 }, { 2, 3 }, { 1, 2 } });
           // int[,] m = new int[,] { { 2, 3, 4 }, { 5, 6, 7 }, { 8, 9, 10 }, { 11, 12, 13 }, { 14, 15, 16 } };
           // s.ImageSmoother(m);
            //s.WidthOfBinaryTree1(new Solution.TreeNode(1)
            //{
            //    left = new Solution.TreeNode(555),
            //    right = new Solution.TreeNode(666)
            //});
            //s.FindComplement(5);
            // s.IsNumber(" -.");
            s.FindDisappearedNumbers(new[] {4, 3, 2, 7, 8, 2, 3, 1});
            Console.ReadKey();
        }
    }

    public class Solution
    {
        public int GetSum(int a, int b)
        {
            // TODO Write a blog for this to ensure understanding this
            if (a == 0) return b;
            if (b == 0) return a;
            while (b != 0)
            {
                int carry = a & b;
                a = a ^ b;
                b = carry << 1;
            }
            return a;
        }

        public int AddDigits(int num)
        {
            if (num == 0) return 0;
            if (num % 9 ==0) return 9;
            return num % 9;
        }

        public IList<int> FindDisappearedNumbers(int[] nums)
        {
            for (var i = 0; i < nums.Length; i++)
            {
                var num = Math.Abs(nums[i]); // get the current num
                var val = nums[num - 1]; // get the val of the index
                if(val>0)
                    nums[num - 1] = -val; 
            }
            IList<int> result =new List<int>();
            for (var i = 0; i < nums.Length; i++)
            {
                if(nums[i]>0) result.Add(i+1);
            }
            return result;
        }
        public IList<IList<string>> FindLadders(string beginWord, string endWord, IList<string> wordList)
        {
            IList<IList<string>> result =new List<IList<string>>();
            if (!wordList.Contains(endWord)) return result;
            // TODO BFS DFS
            return result;
        }

        IList<string> GetNext(string begin, IList<string> wordList)
        {
            var result = new List<string>();
            for(var i =0;i<wordList.Count;i++)
            {
                var word = wordList[i];
                var count = 0;
                for (var j = 0; j < begin.Length; j++)
                {
                    if (begin[j] != word[j]) count++;
                }
                if (count == 1)
                {
                    result.Add(word);
                    wordList.RemoveAt(i);
                    i--;
                }
            }
            return result;
        }
        public bool IsNumber(string s)
        {
            s = s.Trim();
            if (s == string.Empty) return false;
            char[] symbols = new char[]{'1','2','3','4','5','6','7','8','9','0','e','.','+','-'};
            if (s.Contains('.') && s.IndexOf('.') != s.LastIndexOf('.')) return false;
            if (s.Contains('e') && s.IndexOf('e') != s.LastIndexOf('e')) return false;
            if (s.Contains('e') && s.Contains('.') && s.IndexOf('e') < s.IndexOf('.')) return false;

            for (var i = 0; i < s.Length; i++)
            {
                if (!symbols.Contains(s[i])) return false;
                if (s[i] == '+' || s[i] == '-')
                {
                    if (i != 0 && s[i - 1] != 'e') return false;
                }
                if (s[i] == 'e')
                {
                    if (i == 0||i==s.Length-1) return false;
                    if (s[i - 1] == '+' || s[i - 1] == '-' || s[i - 1] == '.') return false;
                    if (s[i + 1] == '+' || s[i + 1] == '-')
                    {
                        if (i + 1 == s.Length - 1) return false;
                    }
                }
                if (s[i] == '.')
                {
                    if (s.Length == 1) return false;
                    if (i == s.Length - 1 && i - 1 >= 0 && (s[i - 1] == '+' || s[i - 1] == '-')) return false;
                    if (i != s.Length - 1 && s[i + 1] == 'e')
                    {
                        if (i == 0) return false;
                        if (s[i - 1] == '+' || s[i - 1] == '-') return false;
                    }
                }
            }
            return true;
        }
        public string ReverseString(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            StringBuilder sb=new StringBuilder();
            for (var i = s.Length - 1; i >= 0; i--)
            {
                sb.Append(s[i]);
            }
            return sb.ToString();
        }
        public string ReverseWords(string s)
        {
            var temp = s.Split(' ');
            var result = new List<string>();
            foreach (var t in temp)
            {
                result.Add(new string(t.Reverse().ToArray()));
            }
            return string.Join(" ", result);
        }
        public string[] FindWords(string[] words)
        {
            var list1 = new List<char> { 'Q','W','E','R','T','Y','U','I','O','P'};
            var list2 = new List<char>{'A','S','D','F','G','H','J','K','L'};
            var list3 = new List<char> { 'Z', 'X', 'C', 'V', 'B', 'N', 'M'};
            var result=new List<string>();
            foreach (var w in words)
            {
                var upperList = w.Select(x => (char) (x & '\xFFDF')).ToList();
                var upper = upperList[0];
                //var upper = (char) (w[0] & '\xFFDF');
                List<char> list=null;
                if (list1.Contains(upper))
                {
                    list = list1;
                }
                else if(list2.Contains(upper))
                {
                    list = list2;
                }
                else if(list3.Contains(upper))
                {
                    list = list3;
                }
                if (list == null) continue;
                bool flag = true;
                for (var i = 1; i < w.Length; i++)
                {
                    if(!list.Contains(upperList[i]))
                    {
                        flag = false; break;
                    }
                }
                if(flag) result.Add(w);
            }
            return result.ToArray();
        }
        public int FindComplement(int num)
        {
            int mask = ~0;
            while ((num & mask) > 0)
            {
                mask <<= 1;
            }

            return ~mask & ~num;
            //var s = Convert.ToString(num, 2);
            //var complement = new string(s.Select(x => x == '1' ? '0' : '1').ToArray());
            //var result =Convert.ToInt32(complement,2);
            //return result;
        }

        public int WidthOfBinaryTree1(TreeNode root)
        {
            int maxWidth = 0;
            int width;
            int h = height(root);
            int i;

            /* Get width of each level and compare 
               the width with maximum width so far */
            for (i = 1; i <= h; i++)
            {
                var list =new List<int?>();
                getWidth(root, i,list);
                for (var j = 0; j < list.Count; j++)
                {
                    if (list[j] != null)
                    {
                        for (var k = list.Count - 1; k > j; k--)
                        {
                            if (list[k] != null)
                            {
                                maxWidth = Math.Max(maxWidth, k - j+1);
                            }
                        }
                    }
                }
                
            }

            return maxWidth;
        }
        /* Get width of a given level */
        void getWidth(TreeNode node, int level, List<int?> list)
        {
            if (node == null)
            { list.Add(null);
                return;
            }
            

            if (level == 1)
            { list.Add(node.val);
            }
            else if (level > 1)
            {
                getWidth(node.left, level - 1, list);
                getWidth(node.right, level - 1, list);
            }
        }

        /* UTILITY FUNCTIONS */

        /* Compute the "height" of a tree -- the number of
         nodes along the longest path from the root node
         down to the farthest leaf node.*/
        int height(TreeNode node)
        {
            if (node == null)
                return 0;
            else
            {
                /* compute the height of each subtree */
                int lHeight = height(node.left);
                int rHeight = height(node.right);

                /* use the larger one */
                return (lHeight > rHeight) ? (lHeight + 1) : (rHeight + 1);
            }
        }

        public int WidthOfBinaryTree(TreeNode root)
        {
            if (root == null) return 0;
            int maxWidth = 0;
            Queue<TreeNode> q= new Queue<TreeNode>();
            q.Enqueue(root);

            while (q.Count!=0)
            {
                int count = q.Count;
                maxWidth = Math.Max(maxWidth, count);
                while (count --> 0)
                {
                    TreeNode temp = q.Dequeue();
                    if (temp.left != null)
                    {
                        q.Enqueue(temp.left);
                    }
                    if (temp.right != null)
                    {
                        q.Enqueue(temp.right);
                    }
                }
            }
            return maxWidth;
        }
        public int[,] ImageSmoother(int[,] M)
        {
            int row = M.GetLength(0);
            int col = M.GetLength(1);
            int[,] result = new int[row, col];
            for (var i = 0; i < row; i++)
            {
                for (var j = 0; j < col; j++)
                {
                    result[i, j] = GetAverage(i, j, M);
                }
            }
            return result;
        }

        public int GetAverage(int x, int y, int[,] M)
        {
            int row = M.GetLength(0);
            int col = M.GetLength(1);
            var sum = 0;
            var num = 0;
            List<Position> lp = new List<Position>()
            {
                new Position(x-1,y-1),
                new Position(x-1,y),
                new Position(x-1,y+1),
                new Position(x,y-1),
                new Position(x,y),
                new Position(x,y+1),
                new Position(x+1,y-1),
                new Position(x+1,y),
                new Position(x+1,y+1)
            };
            foreach (var item in lp)
            {
                if (!(item.X < 0 || item.X >= row || item.Y < 0 || item.Y >= col))
                {
                    sum += M[item.X, item.Y];
                    num++;
                }
            }
            return (int)Math.Floor((double)sum / num);
        }

        class Position
        {
            public int X { get; set; }
            public int Y { get; set; }

            public Position(int x, int y)
            {
                X = x;
                Y = y;
            }
        }
        public int MinSteps(int n)
        {
            if (n == 1) return 0;
            if (n == 2) return 2;
            if (IsSushu(n)) return n;
            return 2 + MinSteps(n / 2);
        }

        public bool IsSushu(int m)
        {
            var k = (int)Math.Sqrt(m);
            int i;
            for (i = 2; i <= k; i++)
                if (m % i == 0)
                    break;
            if (i > k) return true;
            return false;
        }
        public string PredictPartyVictory(string senate)
        {
            int[] temp = new int[senate.Length];
            return Radiant(senate, temp);
        }

        private string Radiant(string senate, int[] temp)
        {
            for (var i = 0; i < senate.Length; i++)
            {
                if (temp[i] == 1) continue;

                if (senate[i] == 'R')
                {
                    int flag = 0;
                    for (var j = i + 1; j < senate.Length; j++)
                    {
                        if (senate[j] == 'D' && temp[j] == 0)
                        {
                            temp[j] = 1;
                            flag = 1;
                            break;
                        }
                    }
                    if (flag == 0)
                    {
                        for (var j = 0; j < i; j++)
                        {
                            if (senate[j] == 'D' && temp[j] == 0)
                            {
                                temp[j] = 1;
                                flag = 1;
                                break;
                            }
                        }
                    }
                    if (flag == 0)
                    {
                        return "Radiant";
                    }
                }
                else if (senate[i] == 'D')
                {
                    int flag = 0;
                    for (var j = i + 1; j < senate.Length; j++)
                    {
                        if (senate[j] == 'R' && temp[j] == 0)
                        {
                            temp[j] = 1;
                            flag = 1;
                            break;
                        }
                    }
                    if (flag == 0)
                    {
                        for (var j = 0; j < i; j++)
                        {
                            if (senate[j] == 'R' && temp[j] == 0)
                            {
                                temp[j] = 1;
                                flag = 1;
                                break;
                            }
                        }
                    }
                    if (flag == 0)
                    {
                        return "Dire";
                    }
                }
            }

            int index = temp.ToList().IndexOf(0);
            var f = senate[index];
            for (var i = 0; i < temp.Length; i++)
            {
                if (temp[i] == 0 && senate[i] != f)
                {
                    return Radiant(senate, temp);
                }
            }
            return f == 'D' ? "Dire" : "Radiant";
        }

        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;
            public TreeNode(int x) { val = x; }
        }

        public List<int> FindDuplicateSubtrees(TreeNode root)
        {
            if (root == null)
                return null;
            Dictionary<String, Dictionary<String, List<TreeNode>>> map = new Dictionary<String, Dictionary<String, List<TreeNode>>>();
            setTraversal(root, map);
            /// in the end every ArrayList in the inner HashMap of map will contain the nodes containing the duplicate subtrees
            return new List<int>();
        }

        public String[] setTraversal(TreeNode root, Dictionary<String, Dictionary<String, List<TreeNode>>> map)
        {
            if (root == null)
                return new String[2];

            String[] left = setTraversal(root.left, map);
            String[] right = setTraversal(root.right, map);
            String[] response = new String[2];

            response[0] = root.val + left[0] + right[0];
            response[1] = left[1] + root.val + right[1];

            Dictionary<String, List<TreeNode>> innerMap;
            if (map.ContainsKey(response[0]))
            {
                innerMap = map[response[0]];
            }
            else
            {
                innerMap = new Dictionary<String, List<TreeNode>>();
                map.Add(response[0], innerMap);
            }
            List<TreeNode> list;
            if (innerMap.ContainsKey(response[1]))
            {
                list = innerMap[response[1]];
            }
            else
            {
                list = new List<TreeNode>();
                innerMap.Add(response[1], list);
            }
            list.Add(root);
            return response;
        }
        public int[] FindErrorNums(int[] nums)
        {
            int[] result = new int[2];

            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] != i + 1)
                {
                    result[0] = nums[i];
                    result[1] = i + 1;
                }

            }
            return result;
        }
        public string ReplaceWords(IList<string> dict, string sentence)
        {
            var temp = sentence.Split(' ');
            var result = (new string[temp.Length]).ToList();
            foreach (var t in temp)
            {
                int[] k = new int[] { -1, int.MaxValue };
                for (int i = 0; i < dict.Count; i++)
                {
                    if (t.StartsWith(dict[i]))
                    {
                        if (dict[i].Length < k[1])
                        {
                            k[0] = i;
                            k[1] = dict[i].Length;
                        }
                    }
                }
                result.Add(k[0] != -1 ? dict[k[0]] : t);
            }
            string s = "";
            foreach (var r in result)
            {
                s += r + " ";
            }
            return s.TrimEnd(' ').TrimStart(' ');
        }
        public int CountSubstrings(string s)
        {
            return 0;
        }
        class Pair
        {
            public int a { get; set; }
            public int b { get; set; }
        }
        public int FindLongestChain(int[,] pairs)
        {
            int n = pairs.Length / 2;
            //为了更好的使用Array的sort，把二维数组存入类的数组
            var tempPairs = new List<Pair>();
            for (var k = 0; k < n; k++)
            {
                Pair p = new Pair();
                p.a = pairs[k, 0];
                p.b = pairs[k, 1];
                tempPairs.Add(p);
            }
            var ps = tempPairs.ToArray();
            Array.Sort<Pair>(ps, new Comparison<Pair>((x, y) => x.a - y.a));

            int i, j, max = 0;
            int[] mcl = new int[n];

            /* Initialize MCL (max chain length) values for all indexes */
            for (i = 0; i < n; i++)
                mcl[i] = 1;

            /* Compute optimized chain length values in bottom up manner */
            for (i = 1; i < n; i++)
                for (j = 0; j < i; j++)
                    if (ps[i].a > ps[j].b && mcl[i] < mcl[j] + 1) //i的这一项的a大于前 面某一项的b，重新给i的max chain length赋值
                        mcl[i] = mcl[j] + 1;

            // mcl[i] now stores the maximum chain length ending with pair i

            /* Pick maximum of all MCL values */
            for (i = 0; i < n; i++)
                if (max < mcl[i])
                    max = mcl[i];

            return max;
        }
        public string SolveEquation(string equation)
        {
            if (string.IsNullOrEmpty(equation) || !equation.Contains("=")) return "No solution";
            string[] s = equation.Replace(" ", "").Split('=');
            string f = s[0];
            string b = s[1];
            var p = GetString(f);
            var sub = GetString(b);
            var result = new int[] { p[0] - sub[0], p[1] - sub[1] };
            var numx = result[0];
            var num = result[1];
            if (numx == 0 && num == 0) return "Infinite solutions";
            else if (numx == 0 && num != 0) return "No solution";
            else
            {
                return "x=" + ((num / numx) * -1);
            }
        }

        public int[] GetString(string s)
        {

            List<string> plusl = new List<string>();
            List<string> subl = new List<string>();
            for (var i = 0; i < s.Length; i++)
            {
                if (s[i] == '-')
                {
                    subl.Add(GetSubString(s, i + 1));
                }
                else if (s[i] == '+')
                {

                    plusl.Add(GetSubString(s, i + 1));
                }
            }
            if (s[0] != '-')
            {
                int index = s.IndexOf('-', 1);
                int indexp = s.IndexOf('+', 1);
                if (index < 0 && indexp < 0) plusl.Add(s);
                else if (index < 0) plusl.Add(s.Substring(0, indexp));
                else if (indexp < 0) plusl.Add(s.Substring(0, index));
                else
                {
                    plusl.Add(s.Substring(0, Math.Min(index, indexp)));
                }
            }
            var p = add(plusl);
            var sub = add(subl);
            return new int[] { p[0] - sub[0], p[1] - sub[1] };
        }

        public string GetSubString(string s, int startindex)
        {
            var result = -1;
            int index = s.IndexOf('-', startindex);
            int indexp = s.IndexOf('+', startindex);
            if (index < 0 && indexp < 0) return s.Substring(startindex);
            else if (index < 0) return s.Substring(startindex, indexp - startindex);
            else if (indexp < 0) return s.Substring(startindex, index - startindex);
            else
            {
                return s.Substring(startindex, Math.Min(index, indexp) - startindex);
            }
        }

        public int[] add(List<string> sl)
        {
            var result = new int[2];
            var num = 0;
            var numx = 0;
            foreach (var s in sl)
            {
                if (s.Contains("x"))
                {
                    if (s.Trim() == "x") numx += 1;
                    else
                    {
                        numx += int.Parse(s.Substring(0, s.Length - 1));
                    }
                }
                else
                {
                    num += int.Parse(s);
                }
            }
            result[0] = numx;
            result[1] = num;
            return result;
        }

        public IList<double> AverageOfLevels(TreeNode root)
        {
            List<double> result = new List<double>();
            if (root == null) return result;
            int depth = GetTreeHeight(root);
            for (var i = 0; i < depth; i++)
            {
                double sum = 0;
                double num = 0;
                GetList(root, i, ref sum, ref num);
                result.Add(sum / num);
            }
            return result;
        }

        public void GetList(TreeNode root, int level, ref double sum, ref double num)
        {
            double result = 0;
            if (null == root || level < 0) return;
            if (0 == level)
            {
                sum += root.val;
                num++;
                return;
            }

            GetList(root.left, level - 1, ref sum, ref num);
            GetList(root.right, level - 1, ref sum, ref num);

        }

        int GetTreeHeight(TreeNode node)
        {
            if (node == null) return 0;
            int leftHeight = GetTreeHeight(node.left);
            int rightHeight = GetTreeHeight(node.right);
            return Math.Max(leftHeight, rightHeight) + 1;
        }
    }

    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x) { val = x; }
    }
}