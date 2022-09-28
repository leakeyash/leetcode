import trees.TreeNode;

import java.util.*;


public class Solution {


    public static void main(String[] args) {
        int[] a = new int[2];
        System.out.println(++a[0]);
        //new Solution().findMedianSortedArrays(new int[] {1,3}, new int[] {2,4});
        //System.out.println(new Solution().canBeTypedWords("leet code", "lt"));
        // System.out.println(new Solution().reformat("ab123"));
        // new Solution().garbageCollection(new String[] {"G","P","GP","GG"}, new int[] {2,4,3});
        // new Solution().checkDistances("aa", new int[]{
        //         1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});
        // new Solution().longestNiceSubarray(new int[]{3,1,5,11,13});

        // new Solution().countDaysTogether("08-15",
        //         "08-18",
        //         "08-16",
        //         "08-19");
        // new Solution().longestContinuousSubstring("abcabe");
        //new Solution().reverseOddLevels(TreeNodeFactory.newBinaryTree(2,3,5,8,13,21,34));

        // new Solution().maxProfitCoolDown(new int[] {1,2,3,0,2});
        // new Solution().transportationHub(new int[][]{{0,3},{1,0},{1,3},{2,0},{3,0},{3,2}});
        // new Solution().longestSubarray(new int[] {1,2,3,3,2,2});
      //  new Solution().goodIndices(new int[] {2,1,1,1,3,4,1}, 2);
        new Solution().isSubsequence("acb", "ahbgdc");
    }
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length <= 3) {
            return res;
        }
        Arrays.sort(nums);
        int length = nums.length;
        for (int i = 0; i < nums.length - 3; i++) {
            if(i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            if((long)nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target) {
                break;
            }
            if((long)nums[i] + nums[length-1] + nums[length-2] + nums[length-3] < target) {
                continue;
            }
            for (int j = i+1; j < length -2; j++) {
                if(j>i+1 && nums[j] == nums[j-1]) {
                    continue;
                }
                if((long)nums[i] + nums[j+1] + nums[j+2] + nums[j] > target) {
                    break;
                }
                if((long)nums[i] + nums[length-1] + nums[length-2] + nums[j] < target) {
                    continue;
                }
                int left = j+1;
                int right = length-1;
                while(left < right) {
                    long sum = (long)nums[i] + nums[j] + nums[left] + nums[right];
                    if(sum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while(left < right && nums[left] == nums[left+1]) {
                            left ++;
                        }
                        left++;
                        while(left < right && nums[right] == nums[right-1]) {
                            right --;
                        }
                        right--;
                    } else if(sum < target) {
                        left ++;
                    } else {
                        right --;
                    }
                }
            }
        }
        return res;
    }
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length <= 2) {
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] > 0) {
                break;
            }
            if(i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (right > left && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return res;
    }
    public boolean isSubsequence(String s, String t) {
        if(s.length() == 0) {
            return true;
        }
        int m = 0;
        int n = 0;
        while(m<s.length() && n < t.length()) {
            if(s.charAt(m) == t.charAt(n)) {
                m ++;
                n++;
            } else {
                n++;
            }
        }
        return m == s.length();
    }
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> s2t = new HashMap<>();
        Map<Character, Character> t2s = new HashMap<>();
        int m = s.length();
        for (int i = 0; i < m; i++) {
            Character sc = s.charAt(i);
            Character tc = t.charAt(i);
            if(s2t.containsKey(sc) && !s2t.get(sc).equals(tc) ||
                    t2s.containsKey(tc) && !t2s.get(tc).equals(sc)) {
                return false;
            }
            s2t.put(sc, tc);
            t2s.put(tc, sc);
        }
        return true;
    }
    public void moveZeroes(int[] nums) {
        int index = 0;
        int nonZeroIndex = 0;
        while(index < nums.length) {
            if(nums[index] != 0) {
                nums[nonZeroIndex] = nums[index];
                nonZeroIndex ++;
            }
            index ++;
        }
        while(nonZeroIndex < nums.length) {
            nums[nonZeroIndex] = 0;
            nonZeroIndex++;
        }
    }
    public int getKthMagicNumber(int k) {
        int[] dp = new int[k+1];
        dp[1] = 1;
        int p3 = 1,p5 =1,p7=1;
        for (int i = 2; i <= k; i++) {
            int num3 = dp[p3] * 3, num5 = dp[p5] * 5, num7 = dp[p7] * 7;
            int num = Math.min(Math.min(num3,num5),num7);
            if(num == num3) {
                p3++;
            }
            if (num == num5) {
                p5++;
            }
            if (num == num7) {
                p7++;
            }
            dp[i] = num;
        }
        return dp[k];
    }
    public int[] twoSumII(int[] numbers, int target) {
        int l = 0;
        int r = numbers.length - 1;
        while(l < r) {
            int sum = numbers[l] + numbers[r];
            if(sum == target) {
                return new int[]{l+1,r+1};
            }
            if(sum > target) {
                r --;
            } else {
                l++;
            }
        }
        return new int[2];
    }
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap();
        for (int i = 0; i < nums.length; i++) {
            if(map.containsKey(nums[i])) {
                return new int[]{i, map.get(nums[i])};
            }
            map.put(target-nums[i], i);
        }
        return new int[2];
    }
    public int maxmiumScore(int[] cards, int cnt) {
        Arrays.sort(cards);
        int sum = 0;
        int index = cards.length - 1;
        while(cnt > 0){
            sum += cards[index--];
            cnt--;
        }
        if(sum % 2 == 0){
            return sum;
        }
        for(int i = index;i >= 0;i--){
            for(int j = index + 1;j < cards.length;j++){
                sum -= cards[j];
                sum += cards[i];
                if(sum % 2 == 0){
                    return sum;
                }
                sum -= cards[i];
                sum += cards[j];
            }
        }
        return 0;
    }
    public int minimumSwitchingTimes(int[][] source, int[][] target) {
        Map<Integer, Integer>  map = new HashMap<>();
        int row = source.length;
        int col = source[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                map.put(source[i][j], map.getOrDefault(source[i][j],0) + 1);
                map.put(target[i][j], map.getOrDefault(target[i][j],0) - 1);
            }
        }
        int right = 0;
        for (Integer item: map.values()) {
            if (item > 0) {
                right +=item;
            }
        }
        return Math.abs(right);
    }
    public boolean isPalindrome(int x) {
        String s = Integer.toString(x);
        for (int i =0, j= s.length() - 1; i <= j;) {
            if(s.charAt(i) != s.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }

    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) {
            return false;
        }
        int[] table = new int[26];
        for (int i = 0; i < s1.length(); i++) {
            table[s1.charAt(i) - 'a']++;
        }
        for (int i = 0; i < s2.length(); i++) {
            table[s2.charAt(i) - 'a']--;
            if (table[s2.charAt(i) - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }
    public int[] missingTwo(int[] nums) {
        int[] t = calSum(nums);
        int x = t[0];
        int y = t[1];
        int[] res = new int[2];
        double sqrt = Math.sqrt(2 * y - x * x);
        res[0] = (x + (int) sqrt)/2;
        res[1] = (x - (int) sqrt)/2;
        return res;
    }

    private int[] calSum(int[] nums){
        int n = nums.length;
        int N = n+2;
        int sum1 =0;
        int sum2 = 0;
        long sum3 = 0;
        long sum4 = 0;
        for(int i=0;i<n;i++){
            sum1+=nums[i];
            sum3+=nums[i]*nums[i];
        }
        for(int i=1;i<=N;i++){
            sum2+=i;
            sum4+=i*i;
        }
        return new int[]{sum2-sum1,(int)(sum4-sum3)};
    }
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int level = 0;
        TreeNode node = root;
        while (node.left != null) {
            level++;
            node = node.left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (exists(root, level, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }
    public boolean exists(TreeNode root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0) {
            if ((bits & k) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }

    public int minDepthBFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Deque<TreeNode> nodes = new LinkedList<>();
        nodes.offer(root);
        int level = 1;
        while(!nodes.isEmpty()) {
            int size = nodes.size();
            for(int i = 0; i < size; i++){
                TreeNode node = nodes.poll();
                if(node.left == null && node.right == null){
                    return level;
                }
                if(node.left != null){
                    nodes.offer(node.left);
                }
                if(node.right != null){
                    nodes.offer(node.right);
                }
            }
            level++;
        }
        return level;
    }
    public int minDepthDFS(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left == null && root.right == null) {
            return 1;
        }

        int minDepth = Integer.MAX_VALUE;
        if (root.left != null) {
            minDepth = Math.min(minDepthDFS(root.left), minDepth);
        }
        if (root.right != null) {
            minDepth = Math.min(minDepthDFS(root.right), minDepth);
        }

        return minDepth + 1;
    }
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right) {
            int mid = (left + right) >>> 1;
            int num = nums[mid];
            if(num == target) {
                return mid;
            } else if(num > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right) {
            int mid = (right - left) / 2 +left;
            int num = nums[mid];
            if(num == target) {
                return mid;
            } else if (num > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }
    public String addBinary(String a, String b) {
        StringBuilder ans = new StringBuilder();

        int n = Math.max(a.length(), b.length()), carry = 0;
        for (int i = 0; i < n; ++i) {
            carry += i < a.length() ? (a.charAt(a.length() - 1 - i) - '0') : 0;
            carry += i < b.length() ? (b.charAt(b.length() - 1 - i) - '0') : 0;
            ans.append((char) (carry % 2 + '0'));
            carry /= 2;
        }

        if (carry > 0) {
            ans.append('1');
        }
        ans.reverse();

        return ans.toString();
    }

    public List<Integer> inorderTraversalIteration(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode node = root;
        while(node!=null || !stack.isEmpty()) {
            while(node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            res.add(node.val);
            node = node.right;
        }
        return res;
    }
    public List<Integer> postorderTraversalIteration(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        TreeNode prev = null;
        while (!deque.isEmpty() || root!=null) {
            while(root!= null) {
                deque.push(root);
                root = root.left;
            }
            root = deque.pop();
            if (root.right == null || root.right == prev) {
                res.add(root.val);
                prev = root;
                root = null;
            } else {
                deque.push(root);
                root = root.right;
            }
        }
        return res;
    }
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        postorder(root, res);
        return res;
    }
    private void postorder(TreeNode node, List<Integer> res) {
        if(node == null) {
            return;
        }
        postorder(node.left, res);
        postorder(node.right, res);
        res.add(node.val);
    }
    public List<Integer> preorderTraversalIteration(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        while (!deque.isEmpty() || root!=null) {
            while(root!= null) {
                res.add(root.val);
                deque.push(root);
                root = root.left;
            }
            root = deque.pop();
            root = root.right;
        }
        return res;
    }
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        preorder(root, res);
        return res;
    }
    private void preorder(TreeNode node, List<Integer> res) {
        if(node == null) {
            return;
        }
        res.add(node.val);
        preorder(node.left, res);
        preorder(node.right, res);
    }
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorder(root, res);
        return res;
    }

    private void inorder(TreeNode node, List<Integer> res) {
        if(node == null) {
            return;
        }
        inorder(node.left, res);
        res.add(node.val);
        inorder(node.right, res);
    }
    public List<Integer> goodIndices(int[] nums, int k) {
        int[] left = new int[nums.length], right = new int[nums.length];
        for (int i = 2; i < nums.length; i++) {
            right[i] = nums[i - 1] > nums[i - 2] ? 0 : right[i - 1] + 1;
        }
        for (int i = nums.length - 3; i >= 0; i--) {
            left[i] = nums[i + 1] > nums[i + 2] ? 0 : left[i + 1] + 1;
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i < nums.length - 1; ++i) {
            if (left[i] >= k - 1 && right[i] >= k - 1) {
                list.add(i);
            }
        }
        return list;
    }
    public int longestSubarray(int[] nums) {
        int max = 1, result = 0, curr = 0;
        for (int num : nums) {
            if (num == max) {
                result = Math.max(result, ++curr);
            } else if (num > max) {
                max = num;
                result = curr = 1;
            } else {
                curr = 0;
            }
        }
        return result;
    }
    public String[] sortPeople(String[] names, int[] heights) {
        Map<Integer, String> map = new TreeMap<>((a,b) -> b-a);
        for (int i = 0; i < heights.length; i++) {
            map.put(heights[i], names[i]);
        }
        return map.values().toArray(new String[0]);
    }
    public int transportationHub(int[][] path) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        Set<Integer> all = new HashSet<>();
        Set<Integer> hasStart = new HashSet<>();
        for (int[] ints : path) {
            for (int j = 0; j < 2; j++) {
                int start = ints[0];
                int end = ints[1];
                all.add(start);
                all.add(end);
                hasStart.add(start);
                Set<Integer> set = map.getOrDefault(end, new HashSet<>());
                set.add(start);
                map.put(end, set);
            }
        }
        int allSize = all.size();
        for (Map.Entry<Integer, Set<Integer>> item: map.entrySet()) {
            if(item.getValue().size() == allSize -1 && !hasStart.contains(item.getKey())) {
                return item.getKey();
            }
        }
        return -1;
    }
    public int temperatureTrend(int[] temperatureA, int[] temperatureB) {
        int n = temperatureA.length;
        int count = 0;
        int max = 0;
        for (int i = 1; i < n; i++) {
            int tmp1 = temperatureA[i] - temperatureA[i-1];
            int tmp2 = temperatureB[i] - temperatureB[i-1];
            if(tmp1 == tmp2 || (tmp1 < 0 && tmp2 <0) || (tmp1 > 0 && tmp2 > 0)) {
                count ++;
                max = Math.max(count, max);
            } else {
                count = 0;
            }
        }
        return max;
    }
    // public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    //
    // }
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num: nums) {
            numSet.add(num);
        }
        int longest = 0;
        for(int num: numSet) {
            if(!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                while(numSet.contains(currentNum+1)) {
                    currentNum ++;
                    currentStreak++;
                }
                longest = Math.max(longest, currentStreak);
            }
        }
        return longest;
    }
    public int minimumLines(int[][] stockPrices) {
        if(stockPrices.length <= 2) {
            return stockPrices.length - 1;
        }
        Arrays.sort(stockPrices, Comparator.comparingInt(a -> a[0]));
        int res = 1;
        for(int i=2 ;i <stockPrices.length; i++) {
            long dx1 = stockPrices[i-1][0] - stockPrices[i-2][0];
            long dy1 = stockPrices[i-1][1] - stockPrices[i-2][1];
            long dx2 = stockPrices[i][0] - stockPrices[i-1][0];
            long dy2 = stockPrices[i][1] - stockPrices[i-1][1];
            if(dx1 * dy2 != dy1 * dx2) {
                ++res;
            }
        }
        return res;
    }
    public int maxProfitIV(int k, int[] prices) {
        int n = prices.length;
        int[] buys = new int[k+1];
        int[] sells = new int[k+1];
        Arrays.fill(buys, -prices[0]);
        Arrays.fill(sells, 0);
        for (int i = 0; i < n; ++i) {
            for(int j = 1; j<k+1; j++) {
                buys[j] = Math.max(buys[j], sells[j-1] - prices[i]);
                sells[j] = Math.max(sells[j], buys[j] + prices[i]);
            }
        }
        return sells[k];
    }
    public int maxProfitCoolDown(int[] prices) {
        int n = prices.length;
        int minPrice = -100000;
        int[][][] dp = new int[n][2][2];
        dp[0][0][0] = 0;
        dp[0][1][0] = -prices[0];
        dp[0][0][1] = 0;
        dp[0][1][1] = minPrice;
        for (int i =1; i< n; i++) {
            dp[i][0][0] = Math.max(dp[i-1][0][0], dp[i-1][0][1]);
            dp[i][1][0] = Math.max(dp[i-1][0][0] - prices[i], dp[i-1][1][0]);
            dp[i][0][1] = dp[i-1][1][0] + prices[i];
            dp[i][1][1] = minPrice;
        }
        return Math.max(dp[n-1][0][0], dp[n-1][0][1]);
    }
    public int[] sumPrefixScoresAC(String[] words) {
        Trie root = new Trie();
        for (String word : words) {
            Trie node = root;
            for (char c : word.toCharArray()) {
                (node = node.computeIfAbsent(c, t -> new Trie())).count++;
            }
        }
        int[] result = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            Trie node = root;
            for (char c : words[i].toCharArray()) {
                result[i] += (node = node.get(c)).count;
            }
        }
        return result;
    }

    private class Trie extends HashMap<Character, Trie> {
        private int count;
    }
    public int[] sumPrefixScores(String[] words) {
        int[] ans = new int[words.length];
        HashMap<String, Integer> store = new HashMap<>();
        HashSet<String> prefixSet = new HashSet<>();
        List<List<String>> prefixs = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            List<String> prefixSingle = new ArrayList<>();
            for (int j = 0; j < words[i].length(); j++) {
                String prefix = words[i].substring(0, words[i].length() - j);
                prefixSet.add(prefix);
                prefixSingle.add(prefix);
            }
            prefixs.add(prefixSingle);
        }
        for (int i = 0; i < words.length; i++) {
            for (String prefix : prefixSet) {
                if (words[i].startsWith(prefix)) {
                    store.put(prefix, store.getOrDefault(prefix, 0) + 1);
                }
            }
        }
        for (int i = 0; i < words.length; i++) {
            List<String> prefixSingle = prefixs.get(i);
            int sum = 0;
            for (String p : prefixSingle) {
                sum += store.getOrDefault(p, 0);
            }
            ans[i] = sum;
        }
        return ans;
    }

    public TreeNode reverseOddLevels(TreeNode root) {
        List<List<TreeNode>> nodes = new ArrayList<>();
        List<TreeNode> row = new ArrayList<>();
        row.add(root);
        nodes.add(row);
        constructNodeList(nodes);
        for (int i = 0; i < nodes.size(); i++) {
            if (i % 2 == 1) {
                List<TreeNode> treeNodes = nodes.get(i);
                List<TreeNode> preNodes = nodes.get(i - 1);
                int m = 0;
                int n = treeNodes.size() - 1;
                while (m < n) {

                    int val = treeNodes.get(m).val;
                    treeNodes.get(m).val = treeNodes.get(n).val;
                    treeNodes.get(n).val = val;
                    m++;
                    n--;
                }
                for (int j = 0; j < preNodes.size(); j++) {
                    preNodes.get(j).left = treeNodes.get(j * 2);
                    preNodes.get(j).right = treeNodes.get(j * 2 + 1);
                }
            }
        }
        return root;
    }

    private void constructNodeList(List<List<TreeNode>> nodes) {
        List<TreeNode> treeNodes = nodes.get(nodes.size() - 1);
        if (treeNodes.get(0).left == null) {
            return;
        }
        List<TreeNode> row1 = new ArrayList<>();
        for (TreeNode node : treeNodes) {
            row1.add(node.left);
            row1.add(node.right);
        }
        nodes.add(row1);
        constructNodeList(nodes);
    }

    public int longestContinuousSubstring(String s) {
        int max = 0;
        for (int i = 0; i < s.length(); ) {
            int current = 1;
            while (i + 1 < s.length()) {
                if (s.charAt(i + 1) - s.charAt(i) == 1) {
                    current++;
                    i++;
                } else {
                    break;
                }
            }
            if (current > max) {
                max = current;
            }
            i++;
        }
        return max;
    }

    public int smallestEvenMultiple(int n) {
        if (n == 1) {
            return 2;
        }
        if (n % 2 == 0) {
            return n;
        }
        return n * 2;
    }

    public int maxProfit2(int[] prices) {
        int sum = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] - prices[i] > 0) {
                sum += prices[i + 1] - prices[i];
            }
        }
        return sum;
    }

    public int countDaysTogether(String arriveAlice, String leaveAlice, String arriveBob, String leaveBob) {
        int[] days = new int[]{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int aliceArrive = getYearDay(arriveAlice, days);
        int aliceLeave = getYearDay(leaveAlice, days);
        int bobArrive = getYearDay(arriveBob, days);
        int bobLeave = getYearDay(leaveBob, days);
        int left = Math.max(aliceArrive, bobArrive);
        int right = Math.min(aliceLeave, bobLeave);
        return Math.max(right - left + 1, 0);
    }

    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        Arrays.sort(players);
        Arrays.sort(trainers);
        int i = 0;
        int j = 0;
        int count = 0;
        while (i < players.length && j < trainers.length) {
            if (trainers[j] >= players[i]) {
                i++;
                j++;
                count++;
                continue;
            }
            while (j < trainers.length && trainers[j] < players[i]) {
                j++;
            }
        }
        return count;
    }

    private int getYearDay(String str, int[] days) {
        String[] split = str.split("-");
        int month = Integer.parseInt(split[0]);
        int day = Integer.parseInt(split[1]);
        int monthDaySum = 0;
        for (int i = 1; i <= month; i++) {
            monthDaySum += days[i - 1];
        }
        monthDaySum += day;
        return monthDaySum;
    }


    public int maxProfit1(int[] prices) {
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else if (prices[i] - min > max) {
                max = prices[i] - min;
            }
        }
        return max;
    }

    class StockSpanner {
        Stack<Integer> prices, weights;

        public StockSpanner() {
            prices = new Stack();
            weights = new Stack();
        }

        public int next(int price) {
            int w = 1;
            while (!prices.isEmpty() && prices.peek() <= price) {
                prices.pop();
                w += weights.pop();
            }
            prices.push(price);
            weights.push(w);
            return w;
        }
    }

    class StockPrice {
        TreeMap<Integer, Integer> prices = new TreeMap<Integer, Integer>();
        int maxTimestamp = -1;
        HashMap<Integer, Integer> stockPriceStore;

        public StockPrice() {
            stockPriceStore = new HashMap<>();
        }

        public void update(int timestamp, int price) {
            maxTimestamp = Math.max(maxTimestamp, timestamp);
            int prevPrice = stockPriceStore.getOrDefault(timestamp, 0);
            stockPriceStore.put(timestamp, price);
            if (prevPrice > 0) {
                prices.put(prevPrice, prices.get(prevPrice) - 1);
                if (prices.get(prevPrice) == 0) {
                    prices.remove(prevPrice);
                }
            }
            prices.put(price, prices.getOrDefault(price, 0) + 1);
        }

        public int current() {
            return stockPriceStore.getOrDefault(maxTimestamp, 0);
        }

        public int maximum() {
            return prices.lastKey();
        }

        public int minimum() {
            return prices.firstKey();
        }
    }

    public int maxProfit(int[] prices) {
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else if (prices[i] - min > max) {
                max = prices[i] - min;
            }
        }
        return max;
    }

    public int robII(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(robRange(nums, 0, nums.length - 2), robRange(nums, 0, nums.length - 2));
    }

    private int robRange(int[] nums, int start, int end) {
        int n = end - start;
        if (n == 1) {
            return nums[start];
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[start];
        for (int i = start + 2; i <= n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n];
    }

    public int rob(int[] nums) {
        int n = nums.length;
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }

    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int tribonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        int f1 = 0, f2 = 0, f3 = 1, r = 1;
        for (int i = 3; i <= n; ++i) {
            f1 = f2;
            f2 = f3;
            f3 = r;
            r = f1 + f2 + f3;
        }
        return r;
    }

    public int fib(int n) {
        if (n < 2) {
            return n;
        }
        int fn1;
        int fn2 = 0, r = 1;
        for (int i = 2; i <= n; ++i) {
            fn1 = fn2;
            fn2 = r;
            r = fn1 + fn2;
        }
        return r;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(result.get(i - 1).get(j - 1) + result.get(i - 1).get(j));
                }
            }
            result.add(row);
        }
        return result;
    }

    public int longestNiceSubarray(int[] nums) {

        HashSet<Integer> set = new HashSet<>();
        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            set.clear();
            String s = Integer.toBinaryString(nums[i]);
            for (int j = 0; j < s.length(); j++) {
                if (s.charAt(j) == '1') {
                    set.add(s.length() - j);
                }
            }
            boolean end = false;
            for (int j = i + 1; j < nums.length; j++) {
                String s1 = Integer.toBinaryString(nums[j]);
                for (int m = 0; m < s1.length(); m++) {
                    if (s1.charAt(m) == '1' && set.contains(s1.length() - m)) {
                        end = true;
                        max = Math.max(max, j - i);
                        j = nums.length;
                        break;
                    } else if (s1.charAt(m) == '1') {
                        set.add(s1.length() - m);
                    }
                }
            }
            if (!end) {
                max = Math.max(max, nums.length - i);
            }
        }
        return max;
    }

    public boolean checkDistances(String s, int[] distance) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            Character ch = s.charAt(i);
            if (map.containsKey(ch)) {
                map.put(ch, i - map.get(ch) - 1);
            } else {
                map.put(ch, i);
            }
        }
        for (int i = 0; i < distance.length; i++) {
            char c = (char) ('a' + i);
            if (map.containsKey(c)) {
                if (map.get(c) != distance[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (Character ch : s.toCharArray()) {
            if (ch == '(' || ch == '{' || ch == '[') {
                stack.push(ch);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                Character pop = stack.pop();
                if (ch == ')' && pop != '(') {
                    return false;
                } else if (ch == '}' && pop != '{') {
                    return false;
                } else if (ch == ']' && pop != '[') {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    public int numSpecial(int[][] mat) {
        int rowCount = mat.length;
        int colCount = mat[0].length;
        int[] rows = new int[rowCount];
        int[] cols = new int[colCount];
        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < colCount; j++) {
                rows[i] += mat[i][j];
                cols[j] += mat[i][j];
            }
        }
        int count = 0;
        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < colCount; j++) {
                if (mat[i][j] == 1 && rows[i] == 1 && cols[j] == 1) {
                    count++;
                }
            }
        }
        return count;
    }

    public boolean findSubarrays(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            int target = nums[i] + nums[i + 1];
            for (int j = i + 1; j < nums.length - 1; j++) {
                if (nums[j] + nums[j + 1] == target) {
                    return true;
                }
            }
        }
        return false;
    }

    public int garbageCollection(String[] garbage, int[] travel) {
        int[] mList = new int[garbage.length];
        int[] pList = new int[garbage.length];
        int[] gList = new int[garbage.length];
        int mLast = 0;
        int pLast = 0;
        int gLast = 0;
        for (int i = 0; i < garbage.length; i++) {
            for (Character ch : garbage[i].toCharArray()) {
                if (ch == 'M') {
                    mList[i] = mList[i] + 1;
                    mLast = i;
                }
                if (ch == 'P') {
                    pList[i] = pList[i] + 1;
                    pLast = i;
                }
                if (ch == 'G') {
                    gList[i] = gList[i] + 1;
                    gLast = i;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i <= mLast; i++) {
            int travelTime = i == 0 ? 0 : travel[i - 1];
            sum += mList[i] + travelTime;
        }
        for (int i = 0; i <= pLast; i++) {
            int travelTime = i == 0 ? 0 : travel[i - 1];
            sum += pList[i] + travelTime;
        }
        for (int i = 0; i <= gLast; i++) {
            int travelTime = i == 0 ? 0 : travel[i - 1];
            sum += gList[i] + travelTime;
        }
        return sum;
    }

    public String removeStars(String s) {
        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch != '*') {
                stack.push(ch);
            } else {
                stack.pop();
            }
        }
        Character[] characters = stack.toArray(new Character[]{});
        StringBuilder result = new StringBuilder();
        for (char c : characters) {
            result.append(c);
        }
        return result.toString();
    }

    public int[] answerQueries(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int[] result = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int sum = 0;
            int count = 0;
            int j = 0;
            while (j < nums.length) {
                sum += nums[j];
                count++;
                j++;
                result[i] = result[i] + 1;
                if (sum > queries[i]) {
                    result[i] = result[i] - 1;
                    break;
                }
            }
        }
        return result;
    }


    public String reformat(String s) {
        int alphaCount = 0, digitCount = 0;
        for (char ch : s.toCharArray()) {
            if (ch >= 'a' && ch <= 'z') {
                alphaCount++;
            } else {
                digitCount++;
            }
        }
        if (Math.abs(alphaCount - digitCount) > 1) {
            return "";
        }


        int currentDigitIndex;
        int currentAlphaIndex;
        if (alphaCount >= digitCount) {
            currentAlphaIndex = 0;
            currentDigitIndex = 1;
        } else {
            currentDigitIndex = 0;
            currentAlphaIndex = 1;
        }
        char[] result = new char[s.length()];
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= 'a' && c <= 'z') {
                result[currentAlphaIndex] = c;
                currentAlphaIndex = currentAlphaIndex + 2;
            } else {
                result[currentDigitIndex] = c;
                currentDigitIndex = currentDigitIndex + 2;
            }
        }
        return new String(result);
    }

    public int missingNumber(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] != 1) {
                return nums[i - 1] + 1;
            }
        }
        return 0;
    }

    public List<String> readBinaryWatch(int turnedOn) {
        List<String> result = new ArrayList<>();
        for (int h = 0; h < 12; ++h) {
            for (int m = 0; m < 60; m++) {
                if (Integer.bitCount(h) + Integer.bitCount(m) == turnedOn) {
                    result.add(h + ":" + (m < 10 ? "0" : "") + m);
                }
            }
        }
        return result;
    }

    public int diagonalSum(int[][] mat) {
        int n = mat.length, sum = 0;
        for (int rowNum = 0; rowNum < n; rowNum++) {
            for (int colNum = 0; colNum < n; colNum++) {
                if (rowNum == colNum || rowNum + colNum == n - 1) {
                    sum += mat[rowNum][colNum];
                }
            }
        }
        return sum;
    }

    public String orderlyQueue(String s, int k) {
        if (k == 1) {
            String smallest = s;
            StringBuilder sb = new StringBuilder(s);
            int n = s.length();
            for (int i = 1; i < n; i++) {
                char c = sb.charAt(0);
                sb.deleteCharAt(0);
                sb.append(c);
                if (sb.toString().compareTo(smallest) < 0) {
                    smallest = sb.toString();
                }
            }
            return smallest;
        } else {
            char[] arr = s.toCharArray();
            Arrays.sort(arr);
            return new String(arr);
        }
    }

    public char repeatedCharacter(String s) {
        int[] temp = new int[26];
        for (char ch : s.toCharArray()) {
            int current = ++temp[ch - 'a'];
            if (current == 2) {
                return ch;
            }
        }
        return s.charAt(0);
    }

    public int balancedStringSplit(String s) {
        int lNum = 0;
        int rNum = 0;
        int count = 0;
        for (char ch : s.toCharArray()) {
            if (ch == 'L') {
                lNum++;
            } else if (ch == 'R') {
                rNum++;
            }
            if (lNum != 0 && lNum == rNum) {
                count++;
                lNum = 0;
                rNum = 0;
            }
        }
        return count;
    }

    public int canBeTypedWords(String text, String brokenLetters) {
        int[] brokens = new int[26];
        for (char ch : brokenLetters.toCharArray()) {
            brokens[ch - 'a'] = 1;
        }
        int num = 0;
        boolean flag = true;
        text = text + " ";
        for (char ch : text.toCharArray()) {
            if (ch == ' ') {
                if (flag) {
                    num++;
                }
                flag = true;
                continue;
            }
            if (brokens[ch - 'a'] == 1) {
                flag = false;
            }
        }
        return num;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int length1 = nums1.length;
        int length2 = nums2.length;
        int[] nums = new int[length1 + length2];
        int i = 0, j = 0, count = -1;
        while (i < length1 && j < length2) {
            if (nums1[i] < nums2[j]) {
                nums[++count] = nums1[i];
                i++;
            } else if (nums1[i] > nums[j]) {
                nums[++count] = nums2[j];
                j++;
            } else {
                nums[++count] = nums1[i];
                nums[++count] = nums2[j];
                ++i;
                ++j;
            }
        }
        while (i < length1) {
            nums[++count] = nums1[i];
            ++i;
        }

        while (j < length2) {
            nums[++count] = nums2[j];
            ++j;
        }

        return nums.length % 2 != 0 ? nums[nums.length / 2] : ((float) nums[nums.length / 2 - 1] + nums[nums.length / 2]) / 2;
    }

    public int lengthOfLongestSubstring(String s) {
        HashSet<Character> windows = new HashSet<>();
        int length = s.length();
        int right = -1;
        int ans = 0;
        for (int i = 0; i < length; i++) {
            if (i != 0) {
                windows.remove(s.charAt(i - 1));
            }
            while (right + 1 < length && !windows.contains(s.charAt(right + 1))) {
                windows.add(s.charAt(right + 1));
                right++;
            }
            ans = Math.max(ans, right - i + 1);
        }
        return ans;
    }

    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;

        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode pre = dummyHead;
        while (pre.next != null && pre.next.next != null) {
            ListNode first = pre.next;
            ListNode second = pre.next.next;
            pre.next = second;
            first.next = second.next;
            second.next = first;
            pre = first;
        }
        return dummyHead.next;
    }

    public boolean hasCycleUsingSet(ListNode head) {
        Set<ListNode> nodeSet = new HashSet<>();
        ListNode node = head;
        while (node != null) {
            if (nodeSet.contains(node)) {
                return true;
            }
            nodeSet.add(node);
            node = node.next;
        }
        return false;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    private static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
}

