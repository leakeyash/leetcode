import archive.trees.TreeNodeFactory;
import helper.BinaryNode;
import helper.ListNode;
import helper.Node;
import helper.TreeNode;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Contest {
    public static void main(String[] args) {
        System.out.println("Hello");
        System.out.println(String.valueOf(-10));
        var res = new Contest()
                .getSubarrayBeauty(new int[]{-1,-2,-3,-4,-5}, 2, 2);
        System.out.println(res);
    }
    public int[] getSubarrayBeauty(int[] nums, int k, int x) {
        int[] result = new int[nums.length - k + 1], count = new int[101];
        for (int i = 0; i < nums.length; i++) {
            count[nums[i] + 50]++;
            if (i >= k - 1) {
                for (int j = 0, c = 0; j < 50 && c < x; j++) {
                    if ((c += count[j]) >= x) {
                        result[i - k + 1] = j - 50;
                    }
                }
                count[nums[i - k + 1] + 50]--;
            }
        }
        return result;
    }
    public int findKthLargest(int[] nums, int k) {
        //java.util.PriorityQueue 默认为最小堆实现
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        //现将k个元素放进优先队列中
        for (int i = 0; i < k; i++) {
            pq.add(nums[i]);
        }

        //数组余下的元素和pq最大的元素进行比较
        for (int i = k; i < nums.length; i++) {
            //如果数组元素比优先队列中最小的元素大的话
            if(!pq.isEmpty() && nums[i] > pq.peek()){
                //优先队列中最小元素出队
                pq.remove();
                //将数组元素放入优先队列中
                pq.add(nums[i]);
            }
        }
        return pq.peek();
    }
    public int sumOfMultiples(int n) {
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            if(i % 3 == 0 || i % 5 ==0 || i% 7 == 0) {
                ans += i;
            }
        }
        return ans;
    }
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        int ans = arrivalTime + delayedTime;
        if(ans < 24) {
            return ans;
        }
        return ans -24;
    }
    public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
        HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], t -> new ArrayList<>()).add(edge[1]);
            map.computeIfAbsent(edge[1], t -> new ArrayList<>()).add(edge[0]);
        }
        int[] count = new int[n];
        for (int[] trip : trips) {
            minimumTotalPrice(trip[0], trip[1], -1, count, map);
        }
        int[] result = minimumTotalPrice(0, -1, price, count, map);
        return Math.min(result[0], result[1]);
    }

    private boolean minimumTotalPrice(int u, int end, int p, int[] count, HashMap<Integer, ArrayList<Integer>> map) {
        if (u == end) {
            count[u]++;
            return true;
        }
        for (int v : map.getOrDefault(u, new ArrayList<>())) {
            if (v != p && minimumTotalPrice(v, end, u, count, map)) {
                count[u]++;
                return true;
            }
        }
        return false;
    }

    private int[] minimumTotalPrice(int u, int p, int[] count, int[] price, HashMap<Integer, ArrayList<Integer>> map) {
        int[] result = { price[u] * count[u], price[u] * count[u] / 2 };
        for (int v : map.getOrDefault(u, new ArrayList<>())) {
            if (v != p) {
                int[] next = minimumTotalPrice(v, u, count, price, map);
                result = new int[] { result[0] + Math.min(next[0], next[1]), result[1] + next[0] };
            }
        }
        return result;
    }
    public int addMinimum(String word) {
        String word1 = word.replaceAll("abc", "x");
        String word2 = word1.replaceAll("ab", "y").replaceAll("ac", "y").replaceAll("bc", "y");
        int ans = 0;
        for (int i = 0; i < word2.length(); i++) {
            if(word2.charAt(i) == 'y') {
                ans+=1;
            } else if(word2.charAt(i) != 'x') {
                ans+=2;
            }
        }
        return ans;
    }
    public int maxDivScore(int[] nums, int[] divisors) {
        int min = Integer.MAX_VALUE;
        int max = -1;
        for (int i = 0; i < divisors.length; i++) {
            int cur = 0;
            for (int j = 0; j < nums.length; j++) {
                if(nums[j] % divisors[i] == 0) {
                    cur++;
                }
            }
            if(cur > max) {
                max = cur;
                min = divisors[i];
            } else if (cur == max && divisors[i] < min) {
                min = divisors[i];
            }
        }
        return min;
    }
    public int[] rowAndMaximumOnes(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        int max = 0;
        int row = 0;
        for (int i = 0; i < m; i++) {
            int cur = 0;
            for (int j = 0; j < n; j++) {
                if(mat[i][j] == 1) {
                    cur+=1;
                }
            }
            if(cur > max) {
                max = cur;
                row = i;
            }
        }
        return new int[]{row, max};
    }
    class Graph extends HashMap<Integer, ArrayList<int[]>> {

        public Graph(int n, int[][] edges) {
            for (int[] edge : edges) {
                addEdge(edge);
            }
        }

        public void addEdge(int[] edge) {
            computeIfAbsent(edge[0], t -> new ArrayList<>()).add(edge);
        }

        public int shortestPath(int node1, int node2) {
            PriorityQueue<int[]> queue = new PriorityQueue<>((o, p) -> o[0] - p[0]);
            queue.offer(new int[] { 0, node1 });
            for (HashSet<Integer> set = new HashSet<>(); !queue.isEmpty();) {
                int[] poll = queue.poll();
                if (poll[1] == node2) {
                    return poll[0];
                } else if (!set.contains(poll[1])) {
                    set.add(poll[1]);
                    for (int[] i : getOrDefault(poll[1], new ArrayList<>())) {
                        queue.offer(new int[] { poll[0] + i[2], i[1] });
                    }
                }
            }
            return -1;
        }
    }
    public TreeNode replaceValueInTree(TreeNode root) {
        Deque<List<TreeNode>> deque = new LinkedList<>();
        deque.offer(List.of(root));
        root.val = 0 ;
        while (!deque.isEmpty()) {
            int size = deque.size();
            Map<Integer, List<TreeNode>>  map = new HashMap<>();
            int sum = 0;
            int[] ssum = new int[size];
            for (int i = 0; i < size; i++) {
                List<TreeNode> poll = deque.poll();
                map.put(i, poll);
                for (int j = 0; j < poll.size(); j++) {
                    sum += poll.get(j).val;
                    ssum[i] += poll.get(j).val;
                    if(poll.get(j).left != null || poll.get(j).right != null) {
                        List<TreeNode> nodes = new ArrayList<>();
                        if(poll.get(j).left != null) {
                            nodes.add(poll.get(j).left);
                        }
                        if(poll.get(j).right != null) {
                            nodes.add(poll.get(j).right);
                        }
                        deque.offer(nodes);
                    }
                }
            }
            for (int i = 0; i < size; i++) {
                List<TreeNode> treeNodes = map.get(i);
                for (int j = 0; j < treeNodes.size(); j++) {
                    treeNodes.get(j).val = sum - ssum[i];
                }
            }
        }
        return root;
    }
    public long[] findPrefixScore(int[] nums) {
        int n = nums.length;
        int max = 0;
        int[] c = new int[n];
        for (int i = 0; i < n; i++) {
            max = Math.max(nums[i], max);
            c[i] = nums[i] + max;
        }
        long[] ans = new long[n];
        ans[0] = c[0];
        for (int i = 1; i < n; i++) {
            ans[i] = ans[i-1] + c[i];
        }
        return ans;
    }
    public int[] findColumnWidth(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            int cur = 0;
            for (int j = 0; j < m; j++) {
                int length = String.valueOf(grid[j][i]).length();
                cur = Math.max(length, cur);
            }
            ans[i] = cur;
        }
        return ans;
    }
    public long[] distance(int[] nums) {
        HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.computeIfAbsent(nums[i], t -> new ArrayList<>()).add(i);
        }
        long[] result = new long[nums.length];
        for (ArrayList<Integer> list : map.values()) {
            long[] sum = new long[list.size() + 1];
            for (int i = 0; i < list.size(); i++) {
                sum[i + 1] = sum[i] + list.get(i);
            }
            for (int i = 0; i < list.size(); i++) {
                result[list.get(i)] = sum[list.size()] - 2 * sum[i] - list.get(i) * (list.size() - 2L * i);
            }
        }
        return result;
    }

    private List<Integer> ddd(List<Integer> integers) {
        List<Integer> res = new ArrayList<>();
        if(integers.size() < 2) {
            return res;
        }
        for (int i = 1; i < integers.size(); i++) {
            res.add(integers.get(i) - integers.get(i-1));
        }
        return res;
    }
    public int diagonalPrime(int[][] nums) {
        int n = nums.length;
        int max = 0;
        for (int i = 0; i < n; i++) {
            int n1 = nums[i][i];
            int n2 = nums[i][n - i - 1];
            if(isPrime(n1)) {
                max = Math.max(max, n1);
            }
            if(isPrime(n2)) {
                max = Math.max(max, n2);
            }

        }
        return max;
    }

    public boolean isPrime(int x) {
        if(x == 1) {
            return false;
        }
        for (int i = 2; i * i <= x; ++i) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    }
    public int miceAndCheese(int[] reward1, int[] reward2, int k) {
        PriorityQueue<int[]> queue = new PriorityQueue<>((a,b) -> (b[0] - a[0]));
        for (int i = 0; i < reward1.length; i++) {
            queue.add(new int[]{reward1[i] - reward2[i], i});
        }
        Set<Integer> indexes = new HashSet<>();
        while (k > 0) {
            int[] poll = queue.poll();
            indexes.add(poll[1]);
            k--;
        }
        int res = 0;
        for (int i = 0; i < reward1.length; i++) {
            if(indexes.contains(i)) {
                res += reward1[i];
            } else {
                res += reward2[i];
            }
        }
        return res;
    }
    public List<List<Integer>> findMatrix(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int pre = nums[0];
        int maxCount = 0;
        int tmp = 1;
        for (int i = 1; i < nums.length;i++) {
            if (nums[i] == pre) {
                tmp++;
            } else {
                maxCount = Math.max(maxCount, tmp);
                tmp = 1;
                pre = nums[i];
            }
        }
        maxCount = Math.max(maxCount, tmp);
        for (int i = 0; i < maxCount; i++) {
            res.add(new ArrayList<>());
        }
        res.get(0).add(nums[0]);
        pre = nums[0];
        tmp = 0;
        for (int i = 1; i < nums.length;i++) {
            if(nums[i] == pre) {
                tmp++;
                res.get(tmp).add(nums[i]);
            } else {
                tmp = 0;
                res.get(tmp).add(nums[i]);
                pre = nums[i];
            }
        }
        return res;
    }
    public int findTheLongestBalancedSubstring(String s) {
        int index = 0;
        int res = 0;
        while (index < s.length()) {
            if(s.charAt(index) == '0') {
                int zero = 0;
                int one = 0;
                while (index < s.length() && s.charAt(index) == '0') {
                    zero++;
                    index++;
                }
                while (index < s.length() && s.charAt(index) == '1') {
                    one++;
                    index++;
                }
                res = Math.max(Math.min(one, zero) * 2, res);
            } else {
                index++;
            }

        }
        return res;
    }
    public int findSmallestInteger(int[] nums, int value) {
        int[] count = new int[value];
        for (int num : nums) {
            count[(num % value + value) % value]++;
        }
        for (int i = 0;; i++) {
            if (--count[i % value] < 0) {
                return i;
            }
        }
    }
    public int beautifulSubsets(int[] nums, int k) {
        Arrays.sort(nums);
        return beautifulSubsets(0, new HashMap<>(), nums, k) - 1;
    }

    private int beautifulSubsets(int index, HashMap<Integer, Integer> map, int[] nums, int k) {
        if (index == nums.length) {
            return 1;
        }
        int count = beautifulSubsets(index + 1, map, nums, k);
        if (map.getOrDefault(nums[index] - k, 0) == 0) {
            map.put(nums[index], map.getOrDefault(nums[index], 0) + 1);
            count += beautifulSubsets(index + 1, map, nums, k);
            map.put(nums[index], map.get(nums[index]) - 1);
        }
        return count;
    }
    public boolean checkValidGrid(int[][] grid) {
        int n = grid.length;
        int cnt = n * n - 1;
        if(grid[0][0] != 0) {
            return false;
        }
        int cur = 1;
        int row = 0;
        int col = 0;
        while (cur <= cnt) {
            boolean match = false;
            int[][] directions = new int[][] {{-2,-1},{-2,1},{-1,2},{-1,-2},{1,2},{2,1},{1,-2},{2,-1}};
            for (int i = 0; i < directions.length; i++) {
                int curR = row + directions[i][0];
                int curC = col + directions[i][1];
                if(curR>=0 && curR < n && curC>=0 && curC < n && grid[curR][curC] == cur) {
                    match = true;
                    row = curR;
                    col = curC;
                    cur ++;
                    break;
                }
            }
            if(!match) {
                return false;
            }
        }
        return true;
    }
    public int[] evenOddBit(int n) {
        String s = Integer.toBinaryString(n);
        int even = 0;
        int odd = 0;
        for (int i = s.length()-1; i >= 0 ; i--) {
            if(s.charAt(i) == '1') {
                if((s.length()-1 - i) % 2 == 0) {
                    even++;
                } else {
                    odd++;
                }
            }
        }
        return new int[]{even,odd};
    }
    public long findScore(int[] nums) {
        TreeSet<Integer> set = new TreeSet<>((o, p) -> nums[o] == nums[p] ? o - p : nums[o] - nums[p]);
        for (int i = 0; i < nums.length; i++) {
            set.add(i);
        }
        long sum = 0;
        for (int i : set) {
            if (nums[i] > 0) {
                sum += nums[i];
                nums[i > 0 ? i - 1 : i] = nums[i < nums.length - 1 ? i + 1 : i] = 0;
            }
        }
        return sum;
    }
    public int maximizeGreatness(int[] nums) {
        Arrays.sort(nums);
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            j += nums[i] > nums[j] ? 1 : 0;
        }
        return j;
    }
    public int distMoney(int money, int children) {
        if (children > money) return -1;
        if (money == children*8) return children;
        if (money > children*8) return children-1;
        if (money == children*8-4) return children-2;
        return (money-children)/7;
    }
    public long beautifulSubarrays(int[] nums) {
        if(nums.length == 1) {
            if(nums[0] == 0) {
                return 1;
            }
            return 0;
        }
        long res = 0;
        for (int i = 0; i < nums.length-1; i++) {
            int cur = nums[i];
            if (cur == 0) {
                res++;
            }
            for (int j = i+1; j < nums.length; j++) {
                cur = cur ^ nums[j];
                if(cur == 0) {
                    res++;
                }
            }
        }
        if(nums[nums.length-1] == 0) {
            res++;
        }

        return res;
    }
    public int maxScore(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        long sum = 0;
        for (int i = nums.length-1; i >=0 ; i--) {
            sum = sum + nums[i];
            if(sum >0) {
                res++;
            }
        }
        return res;
    }
    public int vowelStrings(String[] words, int left, int right) {
        int res = 0;
        Set<Character> characters = new HashSet<>();
        characters.add('a');
        characters.add('e');
        characters.add('i');
        characters.add('o');
        characters.add('u');
        for (int i = left; i <= right; i++) {
            if(characters.contains(words[i].charAt(0)) && characters.contains(words[i].charAt(words[i].length()-1))) {
                res++;
            }
        }
        return res;
    }
    public int findValidSplit(int[] nums) {

        BigInteger[] caches = new BigInteger[nums.length];
        caches[0] = BigInteger.valueOf(nums[0]);
        for (int i = 1; i < nums.length; i++) {
            caches[i] = caches[i-1].multiply(BigInteger.valueOf(nums[i]));
        }
        BigInteger[] right = new BigInteger[nums.length];
        right[nums.length-1] = BigInteger.valueOf(nums[nums.length-1]);
        for (int i = nums.length-2; i >= 0; i--) {
            right[i] = right[i+1].multiply(BigInteger.valueOf(nums[i]));
        }
        BigInteger gcd = BigInteger.valueOf(0);
        for (int i = 0; i <= nums.length-2; i++) {
            BigInteger prefix = caches[i];
            BigInteger suffix = right[i+1];
            if(!gcd.equals(BigInteger.ZERO)) {
                gcd = suffix.gcd(gcd);
                if(gcd.equals(BigInteger.ONE)) {
                    return i;
                }
            } else {
                gcd = prefix.gcd(suffix);
                if(gcd.equals(BigInteger.ONE)) {
                    return i;
                }
            }
        }
        return -1;
    }
    public long kthLargestLevelSum(TreeNode root, int k) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        List<Long> results = new ArrayList<>();
        while (!deque.isEmpty()) {
            int size = deque.size();
            long sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode pop = deque.pop();
                sum+=pop.val;
                if(pop.left!=null) {
                    deque.offer(pop.left);
                }
                if(pop.right!=null) {
                    deque.offer(pop.right);
                }
            }
            results.add(sum);
        }
        if(results.size() < k) {
            return -1;
        }
        return results.stream().sorted((a,b)->Long.compare(b,a)).toList().get(k);
    }
    public int passThePillow(int n, int time) {
        int i = (n - 1) * 2;
        int i1 = time % i;
        if(i1 <= n-1) {
            return i1+1;
        } else {
            return n - (i1 - (n-1));
        }
    }
    public int countWays(int[][] ranges) {
        Arrays.sort(ranges, Comparator.comparingInt(a -> a[0]));
        if(ranges.length == 1) {
            return 2;
        }
        int rowIndex = 0;
        int right = -1;
        for (int i = 0; i < ranges.length; i++) {
            if(ranges[i][0] > right) {
                rowIndex++;
            }
            right = Math.max(right, ranges[i][1]);
        }
        if(rowIndex == 0) {
            return 2;
        }
        int mod = 1000000007;
//        List<Integer> row = new ArrayList<Integer>();
//        row.add(1);
//        for (int i = 1; i <= rowIndex; ++i) {
//            row.add(0);
//            for (int j = i; j > 0; --j) {
//                row.set(j, (row.get(j) % mod + row.get(j - 1) % mod)%mod);
//            }
//        }
//
//        int result = 0;
//        for (int i = 0; i < row.size(); i++) {
//            result = (result + row.get(i) % 1000000007) % mod;
//        }
        int res = 1;
        while (rowIndex >0) {
            res = (res%mod) * 2;
            rowIndex--;
        }
        return (int) (Math.pow(2, rowIndex) % mod);
    }
    public long coloredCells(int n) {
        if(n == 1) {
            return 1;
        }
        long[] dp = new long[n+1];
        dp[1] = 1;
        dp[2] = 1 + 4;
        for (int i = 3; i < n+1; i++) {
            dp[i] = dp[i-1] + 2 * (2L * (i-1));
        }
        return dp[n];
    }
    public int splitNum(int num) {
        String s = Integer.toString(num);
        int[] nums = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            nums[i] = s.charAt(i) - '0';
        }
        Arrays.sort(nums);
        StringBuilder nums1 = new StringBuilder();
        StringBuilder nums2 = new StringBuilder();
        int index = 0;
        while (index < nums.length) {
            nums1.append(nums[index]);
            index++;
            if(index < nums.length) {
                nums2.append(nums[index]);
            }
            index++;
        }
        return Integer.parseInt(nums1.toString()) + Integer.parseInt(nums2.toString());
    }
    public boolean isItPossible(String word1, String word2) {
        int[] cache1 = new int[26];
        int[] cache2 = new int[26];
        int diff1 = 0;
        int diff2 = 0;
        for (int i = 0; i < word1.length(); i++) {
            cache1[word1.charAt(i)-'a']++;
            if(cache1[word1.charAt(i)-'a']==1) {
                diff1++;
            }
        }
        for (int i = 0; i < word2.length(); i++) {
            cache2[word2.charAt(i)-'a']++;
            if(cache2[word2.charAt(i)-'a']==1) {
                diff2++;
            }
        }
        if(Math.abs(diff1-diff2)>2) {
            return false;
        }
        for (int i = 0; i < 26; i++) {
            int curDiff1 = diff1;
            if(cache1[i]>0) {
                cache1[i]--;
                if(cache1[i]==0) {
                    curDiff1--;
                }
                for (int j = 0; j < 26; j++) {
                    int tmpDiff1 = curDiff1;
                    int curDiff2 = diff2;
                    if(cache2[j]>0) {
                        cache1[j]++;
                        cache2[j]--;
                        cache2[i]++;
                        if(cache1[j] == 1) {
                            tmpDiff1++;
                        }
                        if(cache2[j]==0) {
                            curDiff2--;
                        }
                        if(cache2[i]==1) {
                            curDiff2++;
                        }
                        if(tmpDiff1 == curDiff2) {
                            return true;
                        }
                        cache1[j]--;
                        cache2[j]++;
                        cache2[i]--;
                    }
                }
                cache1[i]++;
            }
        }
        return false;
    }
    public long maxKelements(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>((a,b) -> b-a);
        for (int i: nums) {
            queue.offer(i);
        }
        long res = 0;
        for (int i = 0; i < k; i++) {
            Integer poll = queue.poll();
            res += poll;
            queue.offer((int)Math.ceil((double)poll/3));
        }
        return res;
    }
    public int maximumCount(int[] nums) {
        int pos = 0;
        int neg = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                neg ++;
            } else if(nums[i] > 0) {
                pos++;
            }
        }
        return Math.max(pos, neg);
    }
    public int maximumTastiness(int[] price, int k) {
        int left = 0, right = 1000000000;
        for (Arrays.sort(price); left < right;) {
            int mid = (left + right + 1) / 2, count = 1;
            for (int i = 1, prev = 0; i < price.length; i++) {
                if (price[i] - price[prev] >= mid) {
                    count++;
                    prev = i;
                }
            }
            if (count < k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }
    public int takeCharacters(String s, int k) {
        int left = 0, right = s.length() - 1, map[] = new int[3], min = 0;
        for (; left < s.length() && (map[0] < k | map[1] < k | map[2] < k); min = ++left) {
            map[s.charAt(left) - 'a']++;
        }
        for (; left > 0; min = Math.min(min, s.length() - right + left - 1)) {
            for (map[s.charAt(--left) - 'a']--; right >= 0 && map[s.charAt(left) - 'a'] < k; right--) {
                map[s.charAt(right) - 'a']++;
            }
        }
        return map[0] < k | map[1] < k | map[2] < k ? -1 : min;
    }
    public int closetTarget(String[] words, String target, int startIndex) {
        int res = words.length;
        for (int i = 0; i < words.length; i++) {
            if(words[i].equals(target)) {
                int cur;
                if(i >= startIndex) {
                    cur = Math.min(i-startIndex,(startIndex+words.length-i)%words.length);
                } else {
                    cur = Math.min(startIndex-i,(i+words.length-startIndex)%words.length);
                }
                res = Math.min(res, cur);
            }
        }
        return res == words.length ? -1: res;
    }
    public int maxAreaOfIslandDFS(int[][] grid) {
        int max = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    max = Math.max(max, dfsMaxAreaOfIsland(grid, 0, i, j));
                }
            }
        }
        return max;
    }

    private int dfsMaxAreaOfIsland(int[][] grid, int size, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1) {
            return size;
        } else {
            grid[i][j] = 0;
            size++;
            size = dfsMaxAreaOfIsland(grid, size, i + 1, j);
            size = dfsMaxAreaOfIsland(grid, size, i, j + 1);
            size = dfsMaxAreaOfIsland(grid, size, i - 1, j);
            size = dfsMaxAreaOfIsland(grid, size, i, j - 1);
            return size;
        }
    }
    public int[] maxPoints(int[][] grid, int[] queries) {
        int[] res = new int[queries.length];
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < queries.length; i++) {
            res[i] = dfsMaxPoints(grid, 0, 0, 0, queries[i], new boolean[m][n]);
        }
        return res;
    }

    private int dfsMaxPoints(int[][] grid, int size, int x, int y, int target, boolean[][] visited) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || visited[x][y] || grid[x][y] >= target) {
            return size;
        }
        visited[x][y] = true;
        System.out.println(x+":"+y+":"+target);
        size++;
        size = dfsMaxPoints(grid, size,x - 1, y, target,visited);
        size = dfsMaxPoints(grid, size,x + 1, y, target,visited);
        size = dfsMaxPoints(grid, size, x, y - 1, target,visited);
        size = dfsMaxPoints(grid, size, x, y + 1, target,visited);
        return size;
    }
    private void test() {
        Allocator allocator = new Allocator(5);
        allocator.free(4);
        allocator.allocate(9,5);
        allocator.allocate(5,8);
    }
    class Allocator {
        int[] array;
        public Allocator(int n) {
            array = new int[n];
        }

        public int allocate(int size, int mID) {
            int index = 0;
            int cnt = 0;
            int startIndex = index;
            boolean exist = false;
            while (index < array.length) {
                if(array[index] == 0) {
                    cnt++;
                    index++;
                    if(cnt == size) {
                        exist = true;
                        break;
                    }
                } else {
                    index++;
                    startIndex = index;
                    cnt = 0;
                }
            }
            if(!exist) {
                return -1;
            }
            for (int i = startIndex; i < startIndex + size; i++) {
                array[i] = mID;
            }
            return startIndex;
        }

        public int free(int mID) {
            int cnt = 0;
            for (int i = 0; i < array.length; i++) {
                if(array[i] == mID) {
                    array[i] = 0;
                    cnt++;
                }
            }
            return cnt;
        }
    }
    public int longestSquareStreak(int[] nums) {
        Arrays.sort(nums);
        int res = -1;
        Set<Integer> visited = new HashSet<>();
        Set<Integer> numSet = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            numSet.add(nums[i]);
        }
        for (int i = 0; i < nums.length; i++) {
            if(visited.contains(nums[i])) {
                continue;
            }
            int cnt = 0;
            int cur = nums[i];
            visited.add(cur);
            while (cur <= 100000) {
                cur = cur * cur;
                if(numSet.contains(cur)) {
                    cnt+=1;
                    visited.add(cur);
                } else {
                    break;
                }
            }
            if(cnt != 0) {
                res = Math.max(res, cnt+1);
            }
        }
        return res;
    }
    public int deleteGreatestValue(int[][] grid) {
        int res = 0;
        int cur = 0;
        int index = 0;
        for (int i = 0; i < grid.length; i++) {
            Arrays.sort(grid[i]);
        }
        while (index < grid[0].length) {
            cur = 0;
            for (int i = 0; i < grid.length; i++) {
                cur = Math.max(grid[i][index], cur);
            }
            res+=cur;
            index++;
        }
        return res;
    }
    public int maxStarSum(int[] vals, int[][] edges, int k) {
        Map<Integer, PriorityQueue<Integer>> map = new HashMap<>();
        for (int i = 0; i < edges.length; i++) {
            int[] edge = edges[i];
            PriorityQueue<Integer> orDefault = map.getOrDefault(edge[0], new PriorityQueue<>((a,b)->b-a));
            orDefault.offer(vals[edge[1]]);
            map.put(edge[0], orDefault);
            PriorityQueue<Integer> orDefault1 = map.getOrDefault(edge[1], new PriorityQueue<>((a,b)->b-a));
            orDefault1.offer(vals[edge[0]]);
            map.put(edge[1], orDefault1);
        }
        for (int i = 0; i < vals.length; i++) {
            if(!map.containsKey(i)) {
                map.put(i, new PriorityQueue<>());
            }
        }
        int max = Integer.MIN_VALUE;
        for (Map.Entry<Integer, PriorityQueue<Integer>> item: map.entrySet()) {
            int cur = 0;
            int curSum = vals[item.getKey()];
            PriorityQueue<Integer> queue = item.getValue();
            max = Math.max(curSum, max);
            while(!queue.isEmpty() && cur < k) {
                curSum += queue.poll();
                cur ++;
                max = Math.max(curSum, max);
            }
        }
        return max;
    }
    public int maximumValue(String[] strs) {
        int sum = 0;
        for (int i = 0; i < strs.length; i++) {
            int cur = -1;
            for (int j = 0; j < strs[i].length(); j++) {
                if (strs[i].charAt(j) >= 'a' && strs[i].charAt(j) <= 'z') {
                    cur = strs[i].length();
                    break;
                }
            }
            if(cur == -1) {
                cur = Integer.parseInt(strs[i]);
            }
            sum = Math.max(sum, cur);
        }
        return sum;
    }
    public int minScore(int n, int[][] roads) {
        Arrays.sort(roads, Comparator.comparingInt(a -> a[2]));
        int minScore = roads[0][2];
        int min = Integer.MAX_VALUE;
        Map<Integer, List<int[]>> map = new HashMap<>();
        Deque<Integer> deque = new LinkedList<>();
        boolean[] visited = new boolean[n+1];
        for (int j = 0; j < roads.length; j++) {
            List<int[]> ints = map.getOrDefault(roads[j][0], new ArrayList<>());
            ints.add(new int[]{roads[j][1], roads[j][2]});
            map.put(roads[j][0], ints);
            List<int[]> ints2 = map.getOrDefault(roads[j][1], new ArrayList<>());
            ints2.add(new int[]{roads[j][0], roads[j][2]});
            map.put(roads[j][1], ints2);
            if(roads[j][0] == 1) {
                deque.offer(roads[j][1]);
                visited[1] = true;
                min = Math.min(min, roads[j][2]);
            } else if(roads[j][1] == 1) {
                deque.offer(roads[j][0]);
                visited[1] = true;
                min = Math.min(min, roads[j][2]);
            }
        }
        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                Integer pop = deque.pop();

                List<int[]> ints = map.get(pop);
                for (int j = 0; j < ints.size(); j++) {
                    if(!visited[ints.get(j)[0]]) {
                        deque.offer(ints.get(j)[0]);
                        visited[ints.get(j)[0]] = true;
                    }

                    min = Math.min(ints.get(j)[1], min);
                }
                if(min == minScore) {
                    return minScore;
                }
            }
        }
        return min;
    }
    public long dividePlayers(int[] skill) {
        Arrays.sort(skill);
        int len = skill.length;
        int sum = skill[0] + skill[len-1];
        long res = 0;
        for (int i = 0; i < len/2; i++) {
            if(skill[i] + skill[len - i - 1] != sum) {
                return -1;
            }
            res += (long)skill[i] * skill[len-i-1];
        }
        return res;
    }
    public boolean isCircularSentence(String sentence) {
        String[] split = sentence.split(" ");
        if(split.length == 1) {
            return split[0].charAt(0) == split[0].charAt(split[0].length()-1);
        }
        for (int i = 1; i < split.length; i++) {
            if(i == split.length-1) {
                boolean bol = split[i].charAt(split[i].length() - 1) == split[0].charAt(0);
                if(!bol) {
                    return false;
                }
                boolean cur = split[i].charAt(0) == split[i-1].charAt(split[i-1].length() - 1);
                if(!cur) {
                    return false;
                }
            }
        }
        return true;
    }
    public int countSubarrays(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>(Map.of(0, 1));
        int i = 0, curr = 0, count;
        for (; nums[i] != k; i++) {
            map.put(curr += nums[i] < k ? -1 : 1, map.getOrDefault(curr, 0) + 1);
        }
        for (count = map.getOrDefault(curr, 0) + map.getOrDefault(curr - 1, 0); ++i < nums.length;) {
            count += map.getOrDefault(curr += nums[i] < k ? -1 : 1, 0) + map.getOrDefault(curr - 1, 0);
        }
        return count;
    }

    public ListNode removeNodes(ListNode head) {
        Deque<Integer> deque = new LinkedList<>();
        while (head!=null) {
            while (!deque.isEmpty() && deque.peek() < head.val) {
                deque.pop();
            }
            deque.push(head.val);
            head = head.next;
        }
        ListNode dummy = new ListNode(0);
        ListNode tmp = dummy;
        while (!deque.isEmpty()) {
            tmp.next = new ListNode(deque.pollLast());
            tmp = tmp.next;
        }
        return dummy.next;
    }
    public int appendCharacters(String s, String t) {
        int indexS = 0;
        int indexT = 0;
        while (indexT < t.length() && indexS < s.length()) {
            if(s.charAt(indexS) == t.charAt(indexT)) {
                indexT++ ;
                indexS++;
            } else {
                indexS++;
            }
        }
        if(indexT == t.length()) {
            return 0;
        }
        return t.length() - indexT;
    }
    public int pivotInteger(int n) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum+=i;
        }
        int left = 1;
        int right = sum;
        if(left == right) {
            return 1;
        }
        for (int i = 2; i <= n; i++) {
            left += i;
            right -= i-1;
            if(left == right) {
                return i;
            }
        }
        return -1;
    }
    public int countPalindromes(String s) {
        long right[] = new long[10], r[][] = new long[s.length()][100], count = 0;
        for (int i = s.length() - 1; i > 0; right[s.charAt(i--) - '0']++) {
            r[i - 1] = r[i].clone();
            for (int j = 0; j <= 9; j++) {
                r[i - 1][s.charAt(i) - '0' + 10 * j] = (r[i - 1][s.charAt(i) - '0' + 10 * j] + right[j]) % 1000000007;
            }
        }
        for (int i = 0; i <= 99; i++) {
            for (int j = 0, left[] = new int[10], l[] = new int[100]; j < s.length(); left[s.charAt(j++) - '0']++) {
                count = (count + l[i] * r[j][i]) % 1000000007;
                for (int k = 0; k <= 9; k++) {
                    l[s.charAt(j) - '0' + 10 * k] = (l[s.charAt(j) - '0' + 10 * k] + left[k]) % 1000000007;
                }
            }
        }
        return (int) count;

    }
    public int bestClosingTime(String customers) {
        int len = customers.length();
        int least = len+2;
        int index = 0;
        int[] dp = new int[len+1];
        for (int i = 0; i < len; i++) {
            if(customers.charAt(i) == 'Y') {
                dp[0]++;
            }
        }
        for (int i = 1; i < len+1; i++) {
            if(customers.charAt(i-1) == 'Y') {
                dp[i] = dp[i-1] - 1;
            } else {
                dp[i] = dp[i-1] + 1;
            }
        }
        for (int i = 0; i < len; i++) {
            if(dp[i] < least) {
                least = dp[i];
                index = i;
            }
        }
        return index;
    }
    public int[][] onesMinusZeros(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] onesRow = new int[m];
        int[] onesCol = new int[n];
        int[] zerosRow = new int[m];
        int[] zerosCol = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(grid[i][j] == 1) {
                    onesRow[i]++;
                    onesCol[j]++;
                } else {
                    zerosRow[i]++;
                    zerosCol[j]++;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                grid[i][j]= onesRow[i] + onesCol[j] - zerosRow[i] - zerosCol[j];
            }
        }
        return grid;
    }
    public int numberOfCuts(int n) {
        if(n == 1) {
            return 0;
        }
        if(n % 2 == 0) {
            return n/2;
        }
        return n;
    }
    private long result;
    public long minimumFuelCost(int[][] roads, int seats) {
        HashMap<Integer, List<Integer>> map = new HashMap<>();
        for (int[] road : roads) {
            map.computeIfAbsent(road[0], t -> new ArrayList<>()).add(road[1]);
            map.computeIfAbsent(road[1], t -> new ArrayList<>()).add(road[0]);
        }
        for (int i : map.getOrDefault(0, List.of())) {
            minimumFuelCost(i, 0, map, seats);
        }
        return result;
    }

    private int minimumFuelCost(int n, int from, HashMap<Integer, List<Integer>> map, int seats) {
        int count = 1;
        for (int i : map.get(n)) {
            count += i == from ? 0 : minimumFuelCost(i, n, map, seats);
        }
        result += (count + seats - 1) / seats;
        return count;
    }
    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
        List<Integer> list = new ArrayList<>();
        dfsTree(root, list);
        TreeSet<Integer> map = new TreeSet<>(list);
        List<List<Integer>> res = new ArrayList<>();
        for(Integer i: queries) {
            Integer floor = map.floor(i);
            if(floor == null) {
                floor = -1;
            }
            Integer ceiling = map.ceiling(i);
            if(ceiling == null) {
                ceiling = -1;
            }
            res.add(List.of(floor,ceiling));
        }
        return res;
    }

    private void dfsTree(TreeNode root, List<Integer> list) {
        if(root == null) {
            return;
        }
        dfsTree(root.left, list);
        list.add(root.val);
        dfsTree(root.right, list);
    }
    public int unequalTriplets(int[] nums) {
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if(nums[j] != nums[i]) {
                    for (int k = j+1; k < nums.length; k++) {
                        if(nums[k]!=nums[i] && nums[k]!=nums[j]) {
                            cnt++;
                        }
                    }
                }
            }
        }
        return cnt;
    }
    public int maxPalindromes(String s, int k) {
        int[] index = new int[s.length()], dp = new int[s.length() + 1];
        for (int i = 0; i < 2 * s.length(); i++) {
            for (int l = i / 2, r = l + i % 2; l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r); l--, r++) {
                index[r] = Math.max(index[r], r - l + 1 < k ? 0 : l + 1);
            }
        }
        for (int i = 0; i < s.length(); i++) {
            dp[i + 1] = Math.max(dp[i], index[i] > 0 ? 1 + dp[index[i] - 1] : 0);
        }
        return dp[s.length()];
    }

    public boolean isPalindrome(String s) {
        StringBuilder sgood = new StringBuilder();
        int length = s.length();
        for (int i = 0; i < length; i++) {
            char ch = s.charAt(i);
            if (Character.isLetterOrDigit(ch)) {
                sgood.append(Character.toLowerCase(ch));
            }
        }
        StringBuffer sgood_rev = new StringBuffer(sgood).reverse();
        return sgood.toString().equals(sgood_rev.toString());
    }

    public int minimumOperations(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        int sum = 0;
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = deque.poll();
                list.add(poll.val);

                if(poll.left != null) {
                    deque.add(poll.left);
                }
                if(poll.right != null) {
                    deque.add(poll.right);
                }
            }
            sum += count(list);
        }
        return sum;
    }

    private int count(List<Integer> list) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(list);
        for (int i = 0; i < list.size(); i++) {
            map.put(list.get(i),i);
        }
        for (int i = 0; i < list.size(); i++) {
            if(list.get(i).equals(priorityQueue.peek())) {
                priorityQueue.poll();
                continue;
            }
            Integer poll = priorityQueue.poll();
            int index = map.get(poll);
            list.set(index, list.get(i));
            map.put(list.get(i), index);
            list.set(i, poll);
            map.put(poll, i);
            res ++;
        }
        return res;
    }
    public int subarrayLCM(int[] nums, int k) {
        int sum = 0;
        for (int i = 0; i < nums.length;i++) {
            if(nums[i] == k) {
                sum ++;
                int j = i + 1;
                while (j< nums.length && k % nums[j] == 0) {
                    j++;
                }
                sum += j - i - 1;
            } else {
                int j = i + 1;
                int lcm = nums[i];
                while (j< nums.length) {
                    lcm = lcm(lcm,nums[j]);
                    if(lcm == k) {
                        sum++;
                        j++;
                    } else if (lcm < k) {
                        j++;
                    } else {
                        break;
                    }
                }
            }

        }
        return sum;
    }

    public  static  int gcd(int a,int b){
        int min=Math.min(a,b);
        int max=Math.max(a,b);
        a=max;
        b=min;
        while(b>0){
            int c=a%b;
            a=b;
            b=c;
        }
        return a;
    }

    public static int lcm(int a,int b){
        return a*b/gcd(a,b);
    }

    public double[] convertTemperature(double celsius) {
        double kelvin = celsius + 273.15;
        double fahrenheit = celsius * 1.80 + 32.00;
        return new double[]{(double)Math.round(kelvin * 100000)/100000, (double)Math.round(fahrenheit * 100000)/100000};
    }


    public String[] splitMessage(String message, int limit) {
        for (int i = 1, j = 1; i <= message.length(); j += ("" + ++i).length()) {
            if ((3 + ("" + i).length()) * i + j + message.length() <= limit * i) {
                String[] result = new String[i];
                for (int k = 1, m = 0; k <= i; k++) {result[k - 1] = message.substring(m,
                            Math.min(message.length(), m += Math.max(0, limit - 3 - ("" + i + k).length()))) + '<' + k
                            + '/' + i + '>';
                }
                return result;
            }
        }
        return new String[0];
    }
    public int countGoodStrings(int low, int high, int zero, int one) {
        int[] dp = new int[high + 1];
        dp[0] = 1;
        int count = 0;
        for (int i = 1; i <= high; i++) {
            dp[i] = ((i < zero ? 0 : dp[i - zero]) + (i < one ? 0 : dp[i - one])) % 1000000007;
            count = (count + dp[i] * (i < low ? 0 : 1)) % 1000000007;
        }
        return count;
    }
    public int distinctAverages(int[] nums) {
        Arrays.sort(nums);
        Set<Double> res = new HashSet<>();
        for (int i = 0; i < nums.length/2; i++) {
            res.add((double)(nums[i] + nums[nums.length - 1 -i])/2);
        }
        return res.size();
    }
    public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
        Collections.sort(robot);
        Arrays.sort(factory, (o, p) -> o[0] - p[0]);
        long[] dp = new long[robot.size() + 1];
        for (int i = 1; i <= robot.size(); i++) {
            dp[i] = 1000000000000L;
        }
        for (int i = 0; i < factory.length; i++) {
            for (int j = robot.size(); j > 0; j--) {
                for (long k = 1, sum = 0; k <= Math.min(factory[i][1], j); k++) {
                    dp[j] = Math.min(dp[j], dp[j - (int) k] + (sum += Math.abs(factory[i][0] - robot.get(j - (int) k))));
                }
            }
        }
        return dp[robot.size()];
    }
    public long totalCost(int[] costs, int k, int candidates) {
        long res = 0;
        if(costs.length <= candidates * 2) {
            PriorityQueue<Integer> pq = new PriorityQueue<>();
            for (int i = 0; i < costs.length; i++) {
                pq.add(costs[i]);
            }
            for (int i = 0; i < k; i++) {
                res += pq.poll();
            }
            return res;
        }

        if(costs.length > candidates * 2) {
            PriorityQueue<Integer> pq1 = new PriorityQueue<>();
            PriorityQueue<Integer> pq2 = new PriorityQueue<>();
            int leftIndex = candidates;
            int rightIndex = costs.length - candidates - 1;
            for (int i = 0; i < candidates; i++) {
                pq1.add(costs[i]);
                pq2.add(costs[costs.length - 1- i]);
            }
            int n = 0;
            while (n < k) {
                if(!pq1.isEmpty() && !pq2.isEmpty() ) {
                    if(pq1.peek() <= pq2.peek()) {
                        res += pq1.poll();
                        if(leftIndex <= rightIndex) {
                            pq1.add(costs[leftIndex]);
                            leftIndex++;
                        }
                    } else {
                        res += pq2.poll();
                        if(leftIndex <= rightIndex) {
                            pq2.add(costs[rightIndex]);
                            rightIndex--;
                        }
                    }
                }
                else if(!pq1.isEmpty()) {
                    res += pq1.poll();
                } else if(!pq2.isEmpty()){
                    res += pq2.poll();
                }
                n++;
            }
            return res;
        }
        return 0;
    }
    public long maximumSubarraySum(int[] nums, int k) {
        long sum = 0;
        Map<Integer, Integer> set = new HashMap<>();
        long res = 0;
        boolean flag = false;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
            if(set.containsKey(nums[i])) {
                flag = true;
            }
            set.put(nums[i], set.getOrDefault(nums[i],0)+1);
        }
        res = flag ? 0 : sum;
        for (int i = k; i < nums.length; i++) {
            sum -= nums[i-k];
            set.put(nums[i-k], set.get(nums[i-k])-1);
            if(set.get(nums[i-k]) == 0) {
                set.remove(nums[i-k]);
            }
            sum += nums[i];
            set.put(nums[i], set.getOrDefault(nums[i],0)+1);
            if(set.size() == k) {
                res = Math.max(res, sum);
            }
        }
        return res;
    }
    public int[] applyOperations(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            if(nums[i] == nums[i+1]) {
                nums[i] = nums[i] * 2;
                nums[i+1] = 0;
            }
        }
        int[] res = new int[nums.length];
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] != 0) {
                res[index] = nums[i];
                index++;
            }
        }
        return res;
    }
    public long makeIntegerBeautiful(long n, int target) {
        int bits = 0;
        int sum = 0;
        long tmp = n;
        while (tmp != 0) {
            sum+= tmp %10;
            tmp = tmp/10;
            bits++;
        }
        if(sum <= target) {
            return 0;
        }
        long right = (long)Math.pow(10,bits);
        for (long i = n+1; i <= right;) {
            String s = Long.toString(i);
            int index = s.length();
            int cnt = 0;
            for (int j = 0; j < s.length(); j++) {
                int cur = s.charAt(j) - '0';
                cnt += cur;
                if(cnt > target) {
                    if(j == 0) {
                        return right - n;
                    } else {
                       i = (long) ((Long.parseLong(s.substring(0,j)) + 1) * Math.pow(10, index - j));
                       break;
                    }
                }
            }
            if(cnt <= target) {
                return i - n;
            }
            // if(s.charAt(0) - '0' > target) {
            //     return right - n;
            // }
            // if(isLess(i, target)) {
            //     return i - n;
            // }
        }
        return 0;
    }

    private boolean isLess(long n, int target) {
        int sum = 0;
        long tmp = n;
        while (tmp != 0) {
            sum+= tmp %10;
            if(sum> target) {
                return false;
            }
            tmp = tmp/10;
        }
        if(sum <= target) {
            return true;
        }
        return false;
    }
    public List<List<String>> mostPopularCreator(String[] creators, String[] ids, int[] views) {
        Map<String, Long> map = new HashMap<>();
        int n = creators.length;
        long max = 0;
        for (int i = 0; i < n; i++) {
            map.put(creators[i], map.getOrDefault(creators[i],0L)+views[i]);
            max = Math.max(map.get(creators[i]), max);
        }
        Map<String, Pair> topCreators = new HashMap<>();
        for(Map.Entry<String, Long> item: map.entrySet()) {
            if(item.getValue() == max) {
                topCreators.put(item.getKey(), null);
            }
        }
        List<List<String>> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if(topCreators.containsKey(creators[i])) {
                Pair pair = topCreators.get(creators[i]);
                if(pair == null) {
                    topCreators.put(creators[i], new Pair(views[i], ids[i]));
                } else {
                    if(views[i] > pair.view || (views[i] == pair.view && ids[i].compareTo(pair.id) < 0)) {
                        topCreators.put(creators[i], new Pair(views[i], ids[i]));
                    }
                }
            }
        }
        for(Map.Entry<String, Pair> item: topCreators.entrySet()) {
            List<String> record = new ArrayList<>();
            record.add(item.getKey());
            record.add(item.getValue().id);
            res.add(record);
        }
        return res;
    }

    private static class Pair {
        Integer view;
        String id;
        public Pair(Integer view, String id) {
            this.view = view;
            this.id = id;
        }
    }
    public int averageValue(int[] nums) {
        int sum = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] % 2 == 0 && nums[i] % 3 ==0) {
                sum += nums[i];
                count++;
            }
        }
        if(count == 0) {
            return 0;
        }
        return sum / count;
    }
    // TLE Weekly Contest 90
    public int[] secondGreaterElement(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        res[0] = getRight(0, nums);
        for (int i = 1; i < len; i++) {
            if(nums[i] == nums[i-1]) {
                res[i] = res[i-1];
            } else {
                res[i] = getRight(i,nums);
            }
        }
        return res;
    }
    private int getRight(int cur, int[] nums) {
        int right = cur+1;
        int cnt =0;
        while (right<nums.length) {
            if(nums[right] > nums[cur]) {
                cnt++;
                if(cnt == 2) {
                    return nums[right];
                }
            }
            right++;
        }
        return -1;
    }

    public int[] secondGreaterElementRight(int[] nums) {
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            result[i] = -1;
        }
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        PriorityQueue<Integer> queue = new PriorityQueue<>((o, p) -> nums[o] - nums[p]);
        for (int i = 0; i < nums.length; deque.push(i++)) {
            for (; !queue.isEmpty() && nums[queue.peek()] < nums[i]; result[queue.poll()] = nums[i]) {
            }
            for (; !deque.isEmpty() && nums[deque.peek()] < nums[i]; queue.offer(deque.pop())) {
            }
        }
        return result;
    }

    public int destroyTargetsRight(int[] nums, int space) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int max = 0, min = Integer.MAX_VALUE;
        for (int num : nums) {
            map.put(num % space, map.getOrDefault(num % space, 0) + 1);
            max = Math.max(max, map.get(num % space));
        }
        for (int num : nums) {
            min = map.get(num % space) < max ? min : Math.min(min, num);
        }
        return min;
    }

    public int destroyTargets(int[] nums, int space) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int mod = nums[i] % space;
            map.put(mod, map.getOrDefault(mod,0)+1);
        }
        int maxCount = 0;
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if(entry.getValue() > maxCount) {
                maxCount = entry.getValue();
            }
        }
        List<Integer> integers = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if(entry.getValue() == maxCount) {
                integers.add(entry.getKey());
            }
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            int mod = nums[i] % space;
            if(integers.contains(mod)) {
                return nums[i];
            }
        }

        return 0;
    }
    public List<String> twoEditWords(String[] queries, String[] dictionary) {
        int wordLen = queries[0].length();
        if(wordLen <= 2) {
            return Arrays.stream(queries).toList();
        }
        List<String> res = new ArrayList<>();
        for (int i = 0; i < queries.length; i++) {
            for (int j = 0; j < dictionary.length; j++) {
                if(twoSame(queries[i], dictionary[j], wordLen)) {
                    res.add(queries[i]);
                    break;
                }
            }
        }
        return res;

    }

    private boolean twoSame(String a, String b, int len) {
        int diffCount = 0;
        for (int i = 0; i < len; i++) {
            if(a.charAt(i) != b.charAt(i)) {
                diffCount++;
            }
        }
        if(diffCount > 2) {
            return false;
        }
        return true;
    }
    public String oddString(String[] words) {
        HashMap<List<Integer>, List<String>> map = new HashMap<>();
        for (String word : words) {
            map.computeIfAbsent(
                    IntStream.range(1, word.length()).map(o -> word.charAt(o) - word.charAt(o - 1)).boxed().toList(),
                    t -> new ArrayList<>()).add(word);
        }
        for (List<String> list : map.values()) {
            if (list.size() == 1) {
                return list.get(0);
            }
        }
        return "";
    }

}
