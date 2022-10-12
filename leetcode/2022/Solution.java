import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class Solution {

    public static void main(String[] args) {
        System.out.println(Integer.toBinaryString(Integer.MAX_VALUE));
        new Solution().intervalIntersection(new int[][]{{0,2},{5,10},{13,23},{24,25}},new int[][]{{1,5},{8,12},{15,24},{25,26}});
    }
    public int findMin(int[] nums) {
        int l =0, r = nums.length-1;
        while (l<r) {
            int mid = (l+r)>>>1;
            if(nums[mid]>nums[r]) {
                l = mid +1;
            } else {
                r= mid;
            }
        }
        return nums[l];
    }
    public int findPeakElement(int[] nums) {
        if(nums.length==1) {
            return nums[0];
        }
        int l = 0, r= nums.length-1;
        while (l<r) {
            int mid = (l+r)>>>1;
            if (nums[mid] > nums[mid+1]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
    public int maxArea1MS(int[] height) {
        int l = 0, r = height.length - 1;
        int maxArea = 0;
        while (l < r) {
            int area = (r - l) * Math.min(height[l], height[r]);
            maxArea = Math.max(maxArea, area);
            int minH = Math.min(height[l], height[r]);
            while (height[l] <= minH && l < r) {
                l++;
            }
            while (height[r] <= minH && l < r) {
                r--;
            }
        }
        return maxArea;
    }
    public int maxArea(int[] height) {
        int max = 0;
        int left = 0;

        while (left < height.length-1) {
           int right = height.length - 1;
           while(right > left) {
               int preRight = height[right];
               int amount = (right-left) * Math.min(height[left], height[right]);
               max = Math.max(amount, max);
               while(right>left && height[right] <= preRight) {
                   right--;
               }
           }
           int leftH = height[left];
           while(left < height.length-1 && height[left] <= leftH) {
               left++;
           }
        }
        return max;
    }
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        int m = firstList.length;
        int n = secondList.length;
        if(m == 0 || n == 0){
            return new int[][]{};
        }
        List<int[]> res = new ArrayList<>();
        int i =0, j = 0;
        while (i<m && j < n) {
            int fs = firstList[i][0];
            int fe = firstList[i][1];
            int ss = secondList[j][0];
            int se = secondList[j][1];
            int maxS = Math.max(fs, ss);
            int minE = Math.min(se, fe);
            if(minE>=maxS) {
                res.add(new int[]{maxS,minE});
            }
            if(fe > se) {
                j++;
            } else if (se > fe){
                i++;
            } else {
                i++;
                j++;
            }
        }
        return res.toArray(new int[][]{});
    }
    public ListNode deleteDuplicates(ListNode head) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode left = res;
        ListNode right;
        while (left.next != null) {
            right = left.next.next;
            boolean flag = false;
            while(right!= null && right.val == left.next.val) {
                right = right.next;
                flag=true;
            }
            if(flag) {
                left.next = right;
            } else {
                left = left.next;
            }
        }
        return res.next;
    }
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (n==0) {
            return;
        }
        if(m ==0) {
            while (n>0) {
                n = n-1;
                nums1[n] = nums2[n];
            }
            return;
        }

        int i = m-1;
        int j = n-1;
        int k = m+n-1;
        while (i>=0 || j >=0) {
            if(i>=0 && j>=0) {
                if(nums1[i] >= nums2[j]) {
                    nums1[k] = nums1[i];
                    i--;
                } else {
                    nums1[k] = nums2[j];
                    j--;
                }
            } else if(j>=0) {
                nums1[k] = nums2[j];
                j--;
            } else {
                i--;
            }
            k--;
        }
    }
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if(set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
        }
        return false;
    }
    public int numComponents(ListNode head, int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int i:nums) {
            set.add(i);
        }
        int res = 0;
        boolean pre = false;
        while(head!=null) {
            if(set.contains(head.val)) {
                if(!pre) {
                    res++;
                    pre = true;
                }
            } else {
                pre = false;
            }
            head = head.next;
        }
        return res;
    }
    public boolean areAlmostEqual(String s1, String s2) {
        int n = s1.length();
        char nextS2 = '0';
        char nextS1 = '0';
        int count = 0;
        for (int i = 0; i < n; i++) {
            if(s1.charAt(i)!=s2.charAt(i)) {
                if(count == 0) {
                    nextS2 = s1.charAt(i);
                    nextS1 = s2.charAt(i);
                    count++;
                } else if(count == 1) {
                    if(s1.charAt(i) != nextS1 || s2.charAt(i) != nextS2) {
                        return false;
                    }
                    count++;
                } else {
                    return false;
                }
            }
        }
        return count != 1;
    }
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> treeMap = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            treeMap.put(words[i], treeMap.getOrDefault(words[i], 0) +1);
        }
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(treeMap.entrySet());
        entries.sort((a, b) -> {
            if(Objects.equals(b.getValue(), a.getValue())) {
                return a.getKey().compareTo(b .getKey());
            }
            return b.getValue() - a.getValue();
        });
        return entries.subList(0,k).stream().map(Map.Entry::getKey).collect(Collectors.toList());
    }
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((a,b) -> b-a);
        for(int i: stones) {
            priorityQueue.offer(i);
        }
        while (priorityQueue.size()>1) {
            Integer p1 = priorityQueue.poll();
            Integer p2 = priorityQueue.poll();
            int n = Math.abs(p1-p2);
            if(n!=0) {
                priorityQueue.offer(n);
            }
        }
        if(priorityQueue.size() == 0) {
            return 0;
        }
        return priorityQueue.poll();
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[] firstCols = new int[m];
        for (int i = 0; i < m; i++) {
            firstCols[i] = matrix[i][0];
        }
        int l = 0, r = m;
        int rowNum = -1;
        while (l < r) {
            int mid = (r+l)>>>1;
            if(firstCols[mid] == target) {
                rowNum = mid;
                break;
            }
            if(firstCols[mid] > target) {
                r = mid ;
            } else {
                l = mid + 1;
            }
        }
        if(rowNum == -1 && l == 0) {
            return false;
        }
        rowNum = rowNum == -1? l - 1: rowNum;
        int[] row = matrix[rowNum];
         l = 0;
         r = n-1;
         while (l <= r) {
             int mid = (r+l)>>>1;
             if(row[mid] == target) {
                 return true;
             }
             if(row[mid] > target) {
                 r = mid - 1 ;
             } else {
                 l = mid + 1;
             }
         }
        return false;
    }
    public int search33(int[] nums, int target) {
        int n = nums.length;
        int l = 0, r = nums.length ;
        while (l < r) {
            int mid=(r+l)>>>1;
            if(nums[mid] == target) {
                return mid;
            }
            if(nums[mid] >= nums[0]) {
                if(nums[mid] > target && target >= nums[0]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if(nums[mid] < target && target <= nums[n-1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return l < n && nums[l] == target ? l : -1;
    }
    public int[] searchRange(int[] nums, int target) {
        int l=binarySearchRange(nums,target);
        int r=binarySearchRange(nums,target+1);
        if(l==nums.length||nums[l]!=target) {
            return new int[]{-1,-1};
        }
        return new int[]{l,r-1};
    }
    private int binarySearchRange(int[] nums,int target){
        int l=0,r=nums.length;
        while(l<r){
            int mid=(r+l)>>>1;
            if(nums[mid]>=target) {
                r=mid;
            } else {
                l=mid+1;
            }
        }
        return l;
    }
    public String decodeString(String s) {
        Deque<Integer> integerStack = new ArrayDeque<>();
        Deque<StringBuilder> stringStack = new ArrayDeque<>();
        StringBuilder sb = new StringBuilder();
        int number = 0;
        for(char c: s.toCharArray()) {
            if(c == '[') {
                integerStack.push(number);
                stringStack.push(sb);
                number = 0;
                sb = new StringBuilder();
            } else if (c == ']') {
                StringBuilder tmp = stringStack.pop();
                int count = integerStack.pop();
                for(int i = 0; i < count; i++) {
                    tmp.append(sb);
                }
                sb = tmp;
            } else if (c >= '0' && c <= '9') {
                number = number * 10 + c - '0';
            } else {
                sb.append(c);
            }
        }

        return sb.toString();
    }
    public boolean backspaceCompare(String s, String t) {
        Deque<Character> dequeS = new LinkedList<>();
        Deque<Character> dequeT = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if(s.charAt(i) == '#') {
                if(!dequeS.isEmpty()) {
                    dequeS.pop();
                }
            } else {
                dequeS.push(s.charAt(i));
            }
        }
        for (int i = 0; i < t.length(); i++) {
            if(t.charAt(i) == '#') {
                if(!dequeT.isEmpty()) {
                    dequeT.pop();
                }
            } else {
                dequeT.push(t.charAt(i));
            }
        }
        if(dequeS.size() != dequeT.size()) {
            return false;
        }
        while(!dequeS.isEmpty()) {
            if(!dequeS.pop().equals(dequeT.pop())) {
                return false;
            }
        }
        return true;
    }
    public String getHint(String secret, String guess) {
        int countA = 0;
        int[] countS = new int[10];
        int[] countG = new int[10];
        for (int i = 0; i < secret.length(); i++) {
            if(secret.charAt(i) == guess.charAt(i)) {
                countA++;
            } else {
                countS[secret.charAt(i)-'0']++;
                countG[guess.charAt(i)-'0']++;
            }
        }
        int countB = 0;
        for (int i = 0; i < 10; i++) {
            countB+=Math.min(countS[i], countG[i]);
        }
        return countA+"A"+countB+"B";
    }
    public int characterReplacement1(String s, int k) {
        int len = s.length();
        if(len < 2) {
            return len;
        }
        char[] chars = s.toCharArray();
        int left = 0;
        int right = 0;
        int res = 0, maxCount = 0;
        int[] freq = new int[26];
        while (right<len) {
            freq[chars[right] - 'A']++;
            maxCount = Math.max(maxCount, freq[chars[right] - 'A']);
            right++;
            if(right-left>maxCount+k) {
                freq[chars[right] - 'A']--;
                left++;
            }
            res = Math.max(res, right-left);
        }
        return res;
    }
    public int characterReplacement(String s, int k) {
        boolean[] tmp = new boolean[26];
        for(char ch: s.toCharArray()) {
            tmp[ch-'A'] = true;
        }
        int res =0;
        for (int i = 0; i < 26; i++) {
            if(!tmp[i]) {continue;}
            char target = (char)('A' + i);
            int left =0,right=0,cnt =0;
            while (right < s.length()) {
                if(s.charAt(right)!=target) {
                    cnt++;
                }
                while (cnt > k) {
                    if(s.charAt(left)!=target) {
                        cnt--;
                    }
                    left++;
                }
                res = Math.max(res, right-left+1);
                right++;
            }
        }
        return res;
    }
    public List<Integer> findAnagrams(String s, String p) {
        int[] cnt = new int[26];
        for(char c: p.toCharArray()) {
            cnt[c-'a']++;
        }
        int left = 0, right = 0;
        List<Integer> res = new ArrayList<>();
        while(right < s.length()) {
            if(cnt[s.charAt(right) - 'a'] > 0) {
                cnt[s.charAt(right) - 'a']--;
                right++;
                if(right-left == p.length()) {
                    res.add(left);
                }
            } else {
                cnt[s.charAt(left) - 'a'] ++;
                left++;
            }
        }
        return res;
    }
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m+1][n+1];
        for (int i = 1; i <= m ; i++) {
            for (int j = 1; j <= n ; j++) {
                if(i == 1 || j == 1) {
                    dp[i][j] = 1;
                }
            }
        }
        for (int i = 2; i <= m ; i++) {
            for (int j = 2; j <= n ; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m][n];
    }
    public int numIslands(char[][] grid) {
        int[] dx = {1, 0, 0, -1};
        int[] dy = {0, 1, -1, 0};
        int m = grid.length;
        int n = grid[0].length;
        int count = 0;
        Deque<int[]> deque = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if(grid[i][j] == '1' && !visited[i][j]) {
                    deque.offer(new int[]{i,j});
                    visited[i][j] = true;
                    while (!deque.isEmpty()) {
                        int[] pop = deque.poll();
                        for (int k = 0; k < 4; k++) {
                            int mx = pop[0] + dx[k], my = pop[1] + dy[k];
                            if(mx < 0 || mx >= m || my <0 || my>=n) {
                                continue;
                            }
                            if(grid[mx][my] == '1' && !visited[mx][my]) {
                                deque.offer(new int[]{mx, my});
                                visited[mx][my] = true;
                            }
                        }
                    }
                    count++;
                }
            }
        }
        return count;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        while (root != null) {
            if (root.val > p.val && root.val > q.val) {
                root = root.left;
            } else if (root.val < p.val && root.val < q.val) {
                root = root.right;
            } else {
                return root;
            }
        }
        return null;
    }
    public boolean isValidBST(TreeNode root) {
        return dfsIsValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    private boolean dfsIsValidBST(TreeNode node, long low, long high) {
        if(node == null) {
            return true;
        }
        if(node.val <= low || node.val >= high) {
            return false;
        }
        return dfsIsValidBST(node.left, low, node.val) && dfsIsValidBST(node.right, node.val, high);
    }
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) {
            return res;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> row = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = deque.poll();
                row.add(poll.val);
                if(poll.left != null) {
                    deque.offer(poll.left);
                }
                if(poll.right != null) {
                    deque.offer(poll.right);
                }
            }
            res.add(row);
        }
        return res;
    }

    public String robotWithString(String s) {
        Deque<Character> deque = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        doRobotWithString(s.toCharArray(), 0, sb, deque);
        return sb.toString();
    }
    private void doRobotWithString(char[] chars, int startIndex, StringBuilder sb, Deque<Character> deque) {
        if(sb.length() == chars.length) {
            return;
        }
        if(startIndex >= chars.length) {
            while (!deque.isEmpty()) {
                sb.append(deque.pop());
            }
            return;
        }
        int minIndex = startIndex;
        char min = chars[startIndex];
        for (int i = startIndex; i < chars.length; i++) {
            if(chars[i] < min) {
                min = chars[i];
                minIndex = i;
            }
        }
        while(!deque.isEmpty() && min >= deque.peek()) {
           sb.append(deque.pop());
        }
        sb.append(min);

        for (int i = startIndex; i < minIndex; i++) {
            deque.push(chars[i]);
        }
        doRobotWithString(chars, minIndex+1,sb, deque);
    }
    public int[] findArray(int[] pref) {
        int[] res = new int[pref.length];
        res[0]= pref[0];
        for (int i = 1; i < pref.length; i++) {
            res[i] = pref[i] ^ pref[i-1];
        }
        return res;
    }
    public int hardestWorker(int n, int[][] logs) {
        int start = 0;
        int max = 0;
        int id = n;
        for (int i = 0; i < logs.length; i++) {
            int cur = logs[i][1] - start;
            if(cur>max) {
                max = cur;
                id = logs[i][0];
            } else if(cur == max) {
                id = Math.min(id, logs[i][0]);
            }
            start = logs[i][1];
        }
        return id;
    }
    public List<Integer> getRow(int rowIndex) {
        if(rowIndex == 0) {
            return List.of(1);
        }
        List<Integer> res = new ArrayList<>();
        res.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            res.add(Long.valueOf((long)res.get(i-1) * (rowIndex-i+1)/i).intValue());
        }
        return res;
    }
    public int singleNumber(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 1; i <= 32; i++) {
            res = (res << 1) | (n&1);
            n >>>=1;
        }
        return res;
    }
    public boolean find132patternStack(int[] nums) {
        if(nums.length<3){
            return false;
        }
        Deque<Integer> deque = new LinkedList<>();
        deque.push(nums[nums.length-1]);
        int max2 = Integer.MIN_VALUE;
        for (int i = nums.length-2; i >=0 ; i--) {
            if(nums[i] < max2) {
                return true;
            }
            while(!deque.isEmpty() && nums[i] > deque.peek()) {
                max2 = deque.pop();
            }
            deque.push(nums[i]);
        }
        return false;
    }
    public boolean find132pattern(int[] nums) {
        if(nums.length<3){
            return false;
        }
        int left = nums[0];
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int i = 2; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        for (int i = 1; i < nums.length - 1; i++) {
            if(nums[i]>left) {
                Integer ceilingKey = map.ceilingKey(left + 1);
                if(ceilingKey!= null && ceilingKey < nums[i]) {
                    return true;
                }
            }
            left = Math.min(left, nums[i]);
            map.put(nums[i+1], map.get(nums[i+1]) - 1);
            if(map.get(nums[i+1]) == 0) {
                map.remove(nums[i+1]);
            }
        }
        return false;
    }
    public int hammingWeight(int n) {
        int cnt = 0;
        while (n!=0) {
            cnt++;
            n = n&(n-1);
        }
        return cnt;
    }
    public boolean isPowerOfTwo(int n) {
        if(n <= 0) {
            return false;
        }
        return (n & (n-1)) == 0;
    }
    public int minMeetingRoomsPointers(int[][] intervals) {
        if(intervals.length == 1){
            return 1;
        }
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        int startIndex = 0, endIndex = 0, max = 0, cur = 0;
        while (startIndex<intervals.length&&endIndex<intervals.length) {
            if(starts[startIndex] < ends[endIndex]) {
                cur ++;
                startIndex ++;
                max = Math.max(cur, max);
            } else {
                cur --;
                endIndex ++;
            }
        }
        return max;
    }
    public int minMeetingRooms(int[][] intervals) {
        if(intervals.length == 1){
            return 1;
        }
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        PriorityQueue<Integer> endQueue = new PriorityQueue<>();
        endQueue.offer(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            Integer peek = endQueue.peek();
            if(peek!=null && intervals[i][0] >= peek) {
                endQueue.poll();
            }
            endQueue.offer(intervals[i][1]);
        }
        return endQueue.size();
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        doPermute(nums, new ArrayList<>(),0,res, visited);
        return res;
    }

    private void doPermute(int[] nums, List<Integer> path, int depth, List<List<Integer>> res, boolean[] visited) {
        if(depth == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if(!visited[i]) {
                path.add(nums[i]);
                visited[i] = true;
                doPermute(nums, path, depth+1, res, visited);
                path.remove(path.size()-1);
                visited[i]= false;
            }
        }
    }
    public Node connect(Node root) {
        if(root == null) {
            return null;
        }
        if(root.left == null) {
            return root;
        }
        root.left.next = root.right;
        if(root.next != null) {
            root.right.next = root.next.left;
        }
        connect(root.left);
        connect(root.right);
        return root;
    }
    public Node connectBFS(Node root) {
        if(root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node node = queue.poll();
                if(i<size-1) {
                    node.next= queue.peek();
                }
                if(node.left!=null){
                    queue.offer(node.left);
                }
                if(node.right!=null){
                    queue.offer(node.right);
                }
            }
        }
        return root;
    }
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1==null) {
            return root2;
        }
        if(root2 == null) {
            return root1;
        }
        TreeNode res = new TreeNode(root1.val + root2.val);
        res.left = mergeTrees(root1.left, root2.left);
        res.right = mergeTrees(root1.right, root2.right);
        return res;
    }

    public List<String> subdomainVisits(String[] cpdomains) {
        Map<String, Integer> map = new HashMap<>();
        for (String s : cpdomains) {
            String[] split = s.split(" ");
            int count = Integer.parseInt(split[0]);
            String domains = split[1];
            map.put(domains, map.getOrDefault(domains, 0) + count);
            for (int i = 0; i < domains.length(); i++) {
                if (domains.charAt(i) == '.') {
                    map.put(domains.substring(i + 1), map.getOrDefault(domains.substring(i + 1), 0) + count);
                }
            }
        }
        return map.entrySet().stream().map(x -> x.getValue() + " " + x.getKey()).collect(Collectors.toList());
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

    public int maxAreaOfIsland(int[][] grid) {
        int[] dx = {1, 0, 0, -1};
        int[] dy = {0, 1, -1, 0};
        int ans = 0;
        int r = grid.length, c = grid[0].length;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int cur = 0;
                Queue<int[]> queue = new LinkedList<>();
                queue.offer(new int[]{i, j});
                while (!queue.isEmpty()) {
                    int[] poll = queue.poll();
                    int x = poll[0], y = poll[1];
                    if (x < 0 || x >= r || y < 0 || y >= c || grid[x][y] != 1) {
                        continue;
                    }
                    cur++;
                    grid[x][y] = 0;
                    for (int k = 0; k < 4; k++) {
                        int mx = x + dx[k], my = y + dy[k];
                        queue.offer(new int[]{mx, my});
                    }
                }
                ans = Math.max(cur, ans);
            }
        }
        return ans;
    }

    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        int[] dx = {1, 0, 0, -1};
        int[] dy = {0, 1, -1, 0};
        int currColor = image[sr][sc];
        if (currColor == color) {
            return image;
        }
        int r = image.length, c = image[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{sr, sc});
        image[sr][sc] = color;
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0], y = cell[1];
            for (int i = 0; i < 4; i++) {
                int mx = x + dx[i], my = y + dy[i];
                if (mx >= 0 && mx < r && my >= 0 && my < c && image[mx][my] == currColor) {
                    queue.offer(new int[]{mx, my});
                    image[mx][my] = color;
                }
            }
        }
        return image;
    }

    public int minAddToMakeValid(String s) {
        Deque<Character> deque = new LinkedList<>();
        for (char ch : s.toCharArray()) {
            if (ch == '(') {
                deque.push(ch);
            } else {
                Character peek = deque.peek();
                if (peek != null && peek == '(') {
                    deque.pop();
                } else {
                    deque.push(peek);
                }
            }
        }
        return deque.size();
    }

    public boolean checkOnesSegment(String s) {
        boolean flag = true;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '1') {
                if (!flag) {
                    return false;
                }
            } else {
                flag = false;
            }
        }
        return true;
    }

    public int deleteString(String s) {
        //TODO: Wrong answer
        if (s.length() == 1) {
            return 1;
        }
        int count = 0;
        StringBuilder sb = new StringBuilder(s);
        int i;
        for (i = 1; i <= sb.length() / 2; i++) {
            int left = i - 1;
            int right = i;
            StringBuilder sl = new StringBuilder();
            StringBuilder sr = new StringBuilder();
            while (left >= 0 && right < sb.length()) {
                sl.insert(0, sb.charAt(left));
                sr.append(sb.charAt(right));
                if (sl.toString().equals(sr.toString())) {
                    sb.delete(left, i);
                    count++;
                    i = 0;
                    break;
                }
                left--;
                right++;
            }
        }
        return sb.isEmpty() ? count : count + 1;
    }

    public int minimizeXor(int num1, int num2) {
        char[] s1 = Integer.toBinaryString(num1).toCharArray();
        String s2 = Integer.toBinaryString(num2);
        int count2 = 0;
        for (char ch : s2.toCharArray()) {
            if (ch == '1') {
                count2++;
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s1.length; i++) {
            if (s1[i] == '1' && count2 > 0) {
                count2--;
                res.append("1");
            } else {
                res.append("0");
            }
        }
        for (int i = s1.length - 1; i >= 0; i--) {
            if (s1[i] == '0' && count2 > 0) {
                count2--;
                res.insert(i, "1");
                res.deleteCharAt(i + 1);
            }
        }
        while (count2 > 0) {
            res.insert(0, "1");
            count2--;
        }
        return Integer.parseInt(res.toString(), 2);
    }

    public int maxSum(int[][] grid) {
        int max = 0;
        for (int i = 0; i < grid.length - 2; i++) {
            for (int j = 0; j < grid[0].length - 2; j++) {
                int sum = grid[i][j] + grid[i][j + 1] + grid[i][j + 2]
                        + grid[i + 1][j + 1]
                        + grid[i + 2][j] + grid[i + 2][j + 1] + grid[i + 2][j + 2];
                if (sum > max) {
                    max = sum;
                }
            }
        }
        return max;
    }

    public int commonFactors(int a, int b) {
        int min = Math.min(a, b);
        int count = 0;
        for (int i = 1; i <= min; i++) {
            if (a % i == 0 && b % i == 0) {
                count++;
            }
        }
        return count;
    }

    public int xorAllNums(int[] nums1, int[] nums2) {
        int n1 = nums1.length, n2 = nums2.length;
        int ans = 0;
        for (int a : nums1) {
            if ((n2 & 1) == 1) {
                ans ^= a;
            }
        }
        for (int a : nums2) {
            if ((n1 & 1) == 1) {
                ans ^= a;
            }
        }
        return ans;
    }

    class LUPrefix {
        boolean[] cache;
        int lastIndex = -1;

        public LUPrefix(int n) {
            cache = new boolean[n];
        }

        public void upload(int video) {
            cache[video - 1] = true;
            int i = Math.max(lastIndex, 0);
            for (; i < cache.length; i++) {
                if (!cache[i]) {
                    break;
                }
            }
            lastIndex = i - 1;
        }

        public int longest() {
            return lastIndex + 1;
        }
    }

    public boolean equalFrequency(String word) {
        int[] cache = new int[26];
        for (char ch : word.toCharArray()) {
            cache[ch - 'a']++;
        }
        for (char ch : word.toCharArray()) {
            cache[ch - 'a']--;
            boolean test = true;
            int pre = -1;
            for (int i = 0; i < cache.length; i++) {
                if (cache[i] != 0) {
                    if (pre == -1) {
                        pre = cache[i];
                    }
                    if (cache[i] != pre) {
                        test = false;
                    }
                }
            }
            if (test) {
                return true;
            }
            cache[ch - 'a']++;
        }
        return false;
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Deque<Integer> deque = new LinkedList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int j : nums2) {
            Integer peek = deque.peek();
            while (peek != null && peek < j) {
                map.put(peek, j);
                deque.pop();
                peek = deque.peek();
            }
            deque.push(j);
        }
        int[] ans = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            ans[i] = map.getOrDefault(nums1[i], -1);
        }
        return ans;
    }

    public boolean checkInclusion(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        int[] cache = new int[26];
        for (int i = 0; i < n; i++) {
            cache[s1.charAt(i) - 'a']--;
            cache[s2.charAt(i) - 'a']++;
        }
        int diff = 0;
        for (int i : cache) {
            if (i != 0) {
                diff++;
            }
        }
        if (diff == 0) {
            return true;
        }
        for (int i = n; i < m; i++) {
            int out = s2.charAt(i - n) - 'a';
            int in = s2.charAt(i) - 'a';
            if (out == in) {
                continue;
            }
            if (cache[in] == 0) {
                diff++;
            }
            cache[in]++;
            if (cache[in] == 0) {
                diff--;
            }
            if (cache[out] == 0) {
                diff++;
            }
            cache[out]--;
            if (cache[out] == 0) {
                diff--;
            }
            if (diff == 0) {
                return true;
            }
        }
        return false;
    }

    public int longestPalindrome(String s) {
        int[] cache = new int[128];
        for (char ch : s.toCharArray()) {
            if (cache[ch] > 0) {
                cache[ch]--;
            } else {
                cache[ch]++;
            }
        }
        int sum = IntStream.of(cache).sum();
        if (sum >= 1) {
            return s.length() - sum + 1;
        }
        return s.length();
    }

    public String reformatNumber(String number) {
        StringBuilder digits = new StringBuilder();
        for (char ch : number.toCharArray()) {
            if (ch != ' ' && ch != '-') {
                digits.append(ch);
            }
        }
        StringBuilder ans = new StringBuilder();
        int n = digits.length();
        int index = 0;
        while (n > 0) {
            if (n > 4) {
                ans.append(digits.subSequence(index, index + 3)).append("-");
                index += 3;
                n -= 3;
            } else {
                if (n == 4) {
                    ans.append(digits.substring(index, index + 2)).append("-").append(digits.substring(index + 2, index + 4));
                } else {
                    ans.append(digits.substring(index, index + n));
                }
                break;
            }
        }
        return ans.toString();
    }

    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode slow = head;
        ListNode fast = head;
        while (n > 0) {
            fast = fast.next;
            n--;
        }
        ListNode pre = res;
        while (fast != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = slow.next;
        return res.next;
    }

    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (slow != null && fast != null) {
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                break;
            }
            slow = slow.next;
        }
        return slow;
    }

    public int coinChange(int[] coins, int amount) {
        int max = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] == max ? -1 : dp[amount];
    }

    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        int max = 0;
        for (int k : nums) {
            sum += k;
            if (max < k) {
                max = k;
            }
        }
        if (sum % 2 != 0) {
            return false;
        }
        int target = sum / 2;
        if (max > target) {
            return false;
        }
        boolean[][] dp = new boolean[n][target + 1];
        dp[0][nums[0]] = true;
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            for (int j = 0; j <= target; j++) {
                if (j >= num) {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][target];
    }

    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        int max = dp[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            if (dp[i] > max) {
                max = dp[i];
            }
        }
        return max;
    }

    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        boolean[] row = new boolean[matrix.length];
        boolean[] col = new boolean[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = true;
                    col[j] = true;
                }
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    /*//(1,1):代表第一次进入递归函数，并且从第一个口进入，并且记录进入前链表的状态
        merge(1,1): 1->4->5->null, 1->2->3->6->null
        merge(2,2): 4->5->null, 1->2->3->6->null
        merge(3,2): 4->5->null, 2->3->6->null
        merge(4,2): 4->5->null, 3->6->null
        merge(5,1): 4->5->null, 6->null
        merge(6,1): 5->null, 6->null
        merge(7): null, 6->null
                return l2
        l1.next --- 5->6->null, return l1
        l1.next --- 4->5->6->null, return l1
        l2.next --- 3->4->5->6->null, return l2
        l2.next --- 2->3->4->5->6->null, return l2
        l2.next --- 1->2->3->4->5->6->null, return l2
        l1.next --- 1->1->2->3->4->5->6->null, return l1
        public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
            if (l1 == null) {
                return l2;
            }
            else if (l2 == null) {
                return l1;
            }
            else if (l1.val < l2.val) {
                l1.next = mergeTwoLists(l1.next, l2);
                return l1;
            }
            else {
                l2.next = mergeTwoLists(l1, l2.next);
                return l2;
            }
        }
    */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode prehead = new ListNode(-1);

        ListNode prev = prehead;
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                prev.next = list1;
                list1 = list1.next;
            } else {
                prev.next = list2;
                list2 = list2.next;
            }
            prev = prev.next;
        }
        prev.next = list1 == null ? list2 : list1;
        return prehead.next;
    }

    public String reverseWords(String s) {
        char[] ch = s.toCharArray();
        int len = s.length();
        int i = 0, j = 0;
        while (j < len) {
            while (j < len && ch[j] != ' ') {
                j++;
            }
            int k = j - 1;
            while (i < k) {
                swap(ch, i, k);
                i++;
                k--;
            }
            j++;
            i = j;
        }
        return new String(ch);
    }

    private void swap(char[] ch, int i, int j) {
        char temp = ch[i];
        ch[i] = ch[j];
        ch[j] = temp;
    }

    public void reverseString(char[] s) {
        if (s.length == 1) {
            return;
        }
        int l = 0, r = s.length - 1;
        while (l < r) {
            char tmp = s[r];
            s[r] = s[l];
            s[l] = tmp;
            l++;
            r--;
        }
    }

    public boolean isFlipedString(String s1, String s2) {
        return s1.length() == s2.length() && (s1 + s1).contains(s2);
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length <= 3) {
            return res;
        }
        Arrays.sort(nums);
        int length = nums.length;
        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if ((long) nums[i] + nums[length - 1] + nums[length - 2] + nums[length - 3] < target) {
                continue;
            }
            for (int j = i + 1; j < length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if ((long) nums[i] + nums[j + 1] + nums[j + 2] + nums[j] > target) {
                    break;
                }
                if ((long) nums[i] + nums[length - 1] + nums[length - 2] + nums[j] < target) {
                    continue;
                }
                int left = j + 1;
                int right = length - 1;
                while (left < right) {
                    long sum = (long) nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        left++;
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length <= 2) {
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
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
        if (s.length() == 0) {
            return true;
        }
        int m = 0;
        int n = 0;
        while (m < s.length() && n < t.length()) {
            if (s.charAt(m) == t.charAt(n)) {
                m++;
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
            if (s2t.containsKey(sc) && !s2t.get(sc).equals(tc) ||
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
        while (index < nums.length) {
            if (nums[index] != 0) {
                nums[nonZeroIndex] = nums[index];
                nonZeroIndex++;
            }
            index++;
        }
        while (nonZeroIndex < nums.length) {
            nums[nonZeroIndex] = 0;
            nonZeroIndex++;
        }
    }

    public int getKthMagicNumber(int k) {
        int[] dp = new int[k + 1];
        dp[1] = 1;
        int p3 = 1, p5 = 1, p7 = 1;
        for (int i = 2; i <= k; i++) {
            int num3 = dp[p3] * 3, num5 = dp[p5] * 5, num7 = dp[p7] * 7;
            int num = Math.min(Math.min(num3, num5), num7);
            if (num == num3) {
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
        while (l < r) {
            int sum = numbers[l] + numbers[r];
            if (sum == target) {
                return new int[]{l + 1, r + 1};
            }
            if (sum > target) {
                r--;
            } else {
                l++;
            }
        }
        return new int[2];
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return new int[]{i, map.get(nums[i])};
            }
            map.put(target - nums[i], i);
        }
        return new int[2];
    }

    public int maxmiumScore(int[] cards, int cnt) {
        Arrays.sort(cards);
        int sum = 0;
        int index = cards.length - 1;
        while (cnt > 0) {
            sum += cards[index--];
            cnt--;
        }
        if (sum % 2 == 0) {
            return sum;
        }
        for (int i = index; i >= 0; i--) {
            for (int j = index + 1; j < cards.length; j++) {
                sum -= cards[j];
                sum += cards[i];
                if (sum % 2 == 0) {
                    return sum;
                }
                sum -= cards[i];
                sum += cards[j];
            }
        }
        return 0;
    }

    public int minimumSwitchingTimes(int[][] source, int[][] target) {
        Map<Integer, Integer> map = new HashMap<>();
        int row = source.length;
        int col = source[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                map.put(source[i][j], map.getOrDefault(source[i][j], 0) + 1);
                map.put(target[i][j], map.getOrDefault(target[i][j], 0) - 1);
            }
        }
        int right = 0;
        for (Integer item : map.values()) {
            if (item > 0) {
                right += item;
            }
        }
        return Math.abs(right);
    }

    public boolean isPalindrome(int x) {
        String s = Integer.toString(x);
        for (int i = 0, j = s.length() - 1; i <= j; ) {
            if (s.charAt(i) != s.charAt(j)) {
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
        res[0] = (x + (int) sqrt) / 2;
        res[1] = (x - (int) sqrt) / 2;
        return res;
    }

    private int[] calSum(int[] nums) {
        int n = nums.length;
        int N = n + 2;
        int sum1 = 0;
        int sum2 = 0;
        long sum3 = 0;
        long sum4 = 0;
        for (int i = 0; i < n; i++) {
            sum1 += nums[i];
            sum3 += nums[i] * nums[i];
        }
        for (int i = 1; i <= N; i++) {
            sum2 += i;
            sum4 += i * i;
        }
        return new int[]{sum2 - sum1, (int) (sum4 - sum3)};
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
        while (!nodes.isEmpty()) {
            int size = nodes.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = nodes.poll();
                if (node.left == null && node.right == null) {
                    return level;
                }
                if (node.left != null) {
                    nodes.offer(node.left);
                }
                if (node.right != null) {
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
        while (left <= right) {
            int mid = (left + right) >>> 1;
            int num = nums[mid];
            if (num == target) {
                return mid;
            } else if (num > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (right - left) / 2 + left;
            int num = nums[mid];
            if (num == target) {
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
        while (node != null || !stack.isEmpty()) {
            while (node != null) {
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
        while (!deque.isEmpty() || root != null) {
            while (root != null) {
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
        if (node == null) {
            return;
        }
        postorder(node.left, res);
        postorder(node.right, res);
        res.add(node.val);
    }

    public List<Integer> preorderTraversalIteration(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        while (!deque.isEmpty() || root != null) {
            while (root != null) {
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
        if (node == null) {
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
        if (node == null) {
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
        Map<Integer, String> map = new TreeMap<>((a, b) -> b - a);
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
        for (Map.Entry<Integer, Set<Integer>> item : map.entrySet()) {
            if (item.getValue().size() == allSize - 1 && !hasStart.contains(item.getKey())) {
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
            int tmp1 = temperatureA[i] - temperatureA[i - 1];
            int tmp2 = temperatureB[i] - temperatureB[i - 1];
            if (tmp1 == tmp2 || (tmp1 < 0 && tmp2 < 0) || (tmp1 > 0 && tmp2 > 0)) {
                count++;
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
        for (int num : nums) {
            numSet.add(num);
        }
        int longest = 0;
        for (int num : numSet) {
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (numSet.contains(currentNum + 1)) {
                    currentNum++;
                    currentStreak++;
                }
                longest = Math.max(longest, currentStreak);
            }
        }
        return longest;
    }

    public int minimumLines(int[][] stockPrices) {
        if (stockPrices.length <= 2) {
            return stockPrices.length - 1;
        }
        Arrays.sort(stockPrices, Comparator.comparingInt(a -> a[0]));
        int res = 1;
        for (int i = 2; i < stockPrices.length; i++) {
            long dx1 = stockPrices[i - 1][0] - stockPrices[i - 2][0];
            long dy1 = stockPrices[i - 1][1] - stockPrices[i - 2][1];
            long dx2 = stockPrices[i][0] - stockPrices[i - 1][0];
            long dy2 = stockPrices[i][1] - stockPrices[i - 1][1];
            if (dx1 * dy2 != dy1 * dx2) {
                ++res;
            }
        }
        return res;
    }

    public int maxProfitIV(int k, int[] prices) {
        int n = prices.length;
        int[] buys = new int[k + 1];
        int[] sells = new int[k + 1];
        Arrays.fill(buys, -prices[0]);
        Arrays.fill(sells, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 1; j < k + 1; j++) {
                buys[j] = Math.max(buys[j], sells[j - 1] - prices[i]);
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
        for (int i = 1; i < n; i++) {
            dp[i][0][0] = Math.max(dp[i - 1][0][0], dp[i - 1][0][1]);
            dp[i][1][0] = Math.max(dp[i - 1][0][0] - prices[i], dp[i - 1][1][0]);
            dp[i][0][1] = dp[i - 1][1][0] + prices[i];
            dp[i][1][1] = minPrice;
        }
        return Math.max(dp[n - 1][0][0], dp[n - 1][0][1]);
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

    private static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    private static class TreeNode {
        public int val;
        public TreeNode left;
        public TreeNode right;
        TreeNode() {}
        public TreeNode(int val) { this.val = val; }
        public TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

}

