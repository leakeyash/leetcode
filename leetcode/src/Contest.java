import archive.trees.TreeNodeFactory;
import helper.BinaryNode;
import helper.ListNode;
import helper.Node;
import helper.TreeNode;

import java.math.BigDecimal;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Contest {
    public static void main(String[] args) {
        System.out.println("Hello");
        var res = new Contest().countSubarrays(new int[]{5,19,11,15,13,16,4,6,2,7,10,8,18,20,1,3,17,9,12,14}, 6);
        System.out.println(res);
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
