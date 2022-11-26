import helper.ListNode;
import helper.Node;
import helper.TreeNode;

import java.util.*;

public class Solution {
    public static void main(String[] args) {
        System.out.println("test");
        var res = new Solution()
                .pathSum(TreeNode.newBinaryTree(1000000000,1000000000,null,294967296,null,1000000000,null,1000000000,null,1000000000),0);
        System.out.println(res);
    }
    public int pathSum(TreeNode root, int targetSum) {
        if(root == null) {
            return 0;
        }
        int res = 0;
        res += rootSum(root, targetSum);
        res += pathSum(root.left, targetSum);
        res += pathSum(root.right, targetSum);
        return res;
    }

    public int rootSum(TreeNode node, long targetSum) {
        if(node == null) {
            return 0;
        }
        int res = 0;
        if(node.val == targetSum) {
            res ++;
        }
        res += rootSum(node.left, targetSum - node.val);
        res += rootSum(node.right, targetSum - node.val);
        return res;
    }
    int maxDepth = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return maxDepth;
    }

    private int depth(TreeNode node) {
        if(node == null) {
            return 0;
        }
        int left = depth(node.left);
        int right = depth(node.right);
        maxDepth = Math.max(left+right, maxDepth);
        return Math.max(left, right) + 1;
    }
    public int longestPalindrome(String[] words) {
        int sum = 0;
        boolean[] used = new boolean[words.length];
        boolean single = false;
        for (int i = 0; i < words.length; i++) {
            if(used[i]) {
                continue;
            }
            for (int j = i+1; j < words.length; j++) {
                if(!used[j] && words[i].charAt(0) == words[j].charAt(1) && words[i].charAt(1) == words[j].charAt(0)) {
                    sum += 4;
                    used[i] = true;
                    used[j] = true;
                    break;
                }
            }
            if(!used[i] && words[i].charAt(0) == words[i].charAt(1)) {
                single = true;
            }
        }
        return sum + (single?2:0);
    }
    public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }
    public ListNode sortList(ListNode head, ListNode tail) {
        if (head == null) {
            return null;
        }
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode slow = head, fast = head;
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }
        ListNode mid = slow;
        ListNode list1 = sortList(head, mid);
        ListNode list2 = sortList(mid, tail);
        ListNode sorted = merge(list1, list2);
        return sorted;
    }

    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1!=null && temp2!=null){
            if(temp1.val<= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if(temp1!=null) {
            temp.next = temp1;
        } else if(temp2!=null) {
            temp.next = temp2;
        }
        return dummyHead.next;
    }
    public ListNode oddEvenList(ListNode head) {
        if(head==null || head.next == null) {
            return head;
        }

        ListNode evenHead = head.next;
        ListNode odd = head, even = evenHead;
        while (even!=null && even.next!=null) {
            odd.next= even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }
    public boolean isPalindrome(ListNode head) {
        if(head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next!=null) {
            slow= slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow;
        ListNode pre = null;
        while (tmp!=null) {
            ListNode next = tmp.next;
            tmp.next = pre;
            pre = tmp;
            tmp = next;
        }
        while (pre != null) {
            if(pre.val != head.val) {
                return false;
            }
            head = head.next;
            pre = pre.next;
        }
        return true;
    }
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = head;
        ListNode fast = head;
        while (n > 0) {
            fast = fast.next;
            n--;
        }
        while (fast!= null) {
            dummy = dummy.next;
            slow = slow.next;
            fast = fast.next;
        }
        if(head == slow) {
            return slow.next;
        }
        dummy.next = slow.next;
        return head;
    }
    public String multiply(String num1, String num2) {
        int m = num1.length();
        int n = num2.length();
        if(num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int[] cache = new int[m+n];
        for (int i = 0; i < m; i++) {
            int a = num1.charAt(m - i - 1) - '0';
            int index = 0;
            int carry = 0;
            while (index < n) {
                int b = num2.charAt(n - 1 -index) - '0';
                int t = a * b + carry;
                int resIndex = i + index;
                int resCur = cache[resIndex] + t;
                carry = resCur /10;
                cache[resIndex] = resCur % 10;
                index++;
            }
            while (carry != 0) {
                int resCur = cache[i+index] + carry;
                carry = resCur /10;
                cache[i+index] = resCur % 10;
                index++;
            }
        }
        StringBuilder sb = new StringBuilder();
        boolean flag = false;
        for (int i = cache.length-1; i >= 0; i--) {
            if(cache[i]!=0) {
                flag = true;
                sb.append(cache[i]);
            }
            if(cache[i] == 0) {
                if(flag) {
                    sb.append(cache[i]);
                }
            }
        }
        return sb.toString();
    }
    public String longestCommonPrefix(String[] strs) {
        String first = strs[0];
        if(first.length() == 0) {
            return "";
        }
        if(strs.length == 1) {
            return first;
        }
        int index = 0;
        while (index < first.length()) {
            boolean flag = false;
            for (int i = 1; i < strs.length; i++) {
                if(index >= strs[i].length() || strs[i].charAt(index) != first.charAt(index)) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                break;
            }
            index++;
        }
        return first.substring(0,index);
    }
    public int[] swapNumbers(int[] numbers) {
        numbers[0] = numbers[0] + numbers[1];
        numbers[1] = numbers[0] - numbers[1];
        numbers[0] = numbers[0] - numbers[1];
        return numbers;
    }
    public int[] findBall(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            int index = i;
            int row = 0;
            while (row < m) {
                if (grid[row][index] == 1) {
                    if (index == n - 1 || grid[row][index + 1] == -1) {
                        break;
                    } else {
                        index++;
                        row++;
                    }
                } else if (grid[row][index] == -1) {
                    if (index == 0 || grid[row][index - 1] == 1) {
                        break;
                    } else {
                        index--;
                        row++;
                    }
                }
            }
            res[i] = row == m ? index : -1;
        }
        return res;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int leftEdge = 0, rightEdge = matrix[0].length - 1, highEdge = 0, lowEdge = matrix.length - 1;
        List<Integer> res = new ArrayList<>();
        int num = matrix.length * matrix[0].length;
        while (num > 0) {
            for (int i = leftEdge; i <= rightEdge && num > 0; i++) {
                res.add(matrix[highEdge][i]);
                num--;
            }
            highEdge++;
            for (int i = highEdge; i <= lowEdge && num > 0; i++) {
                res.add(matrix[i][rightEdge]);
                num--;
            }
            rightEdge--;
            for (int i = rightEdge; i >= leftEdge && num > 0; i--) {
                res.add(matrix[lowEdge][i]);
                num--;
            }
            lowEdge--;
            for (int i = lowEdge; i >= highEdge && num > 0; i--) {
                res.add(matrix[i][leftEdge]);
                num--;
            }
            leftEdge++;
        }
        return res;
    }

    public int jump(int[] nums) {
        int right = nums.length - 1;
        int steps = 0;
        while (right > 0) {
            for (int i = 0; i < right; i++) {
                if (i + nums[i] >= right) {
                    right = i;
                    steps++;
                    break;
                }
            }
        }
        return steps;
    }

    public boolean canJump(int[] nums) {
        int maxRight = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > maxRight) {
                return false;
            }
            if (maxRight >= nums.length - 1) {
                return true;
            }
            maxRight = Math.max(maxRight, i + nums[i]);
        }
        return true;
    }

    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        } else if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }
        return Math.max(robRange(nums, 0, nums.length - 2), robRange(nums, 1, nums.length - 1));

    }

    private int robRange(int[] nums, int start, int end) {
        int[] dp = new int[end + 1];
        dp[start] = nums[start];
        dp[start + 1] = Math.max(nums[start], nums[start + 1]);
        for (int i = start + 2; i <= end; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[end];
    }

    public boolean exist(char[][] board, String word) {
        int h = board.length, w = board[0].length;
        boolean[][] visited = new boolean[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                boolean flag = check(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean check(char[][] board, boolean[][] visited, int i, int j, String s, int k) {
        if (board[i][j] != s.charAt(k)) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        boolean result = false;
        for (int[] dir : directions) {
            int ni = i + dir[0], nj = j + dir[1];
            if (ni >= 0 && ni < board.length && nj >= 0 && nj < board[0].length && !visited[ni][nj]) {
                boolean flag = check(board, visited, ni, nj, s, k + 1);
                if (flag) {
                    result = true;
                    break;
                }
            }
        }
        visited[i][j] = false;
        return result;
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        Arrays.sort(candidates);
        boolean[] used = new boolean[candidates.length];
        dfsCombinationSum2(candidates, target, res, path, 0, used);
        return res;
    }

    public void dfsCombinationSum2(int[] candidates, int target, List<List<Integer>> res, List<Integer> path, int start, boolean[] used) {

        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (used[i] || (i > start && candidates[i] == candidates[i - 1] && !used[i - 1])) {
                continue;
            }
            if (target - candidates[i] < 0) {
                continue;
            }
            path.add(candidates[i]);
            used[i] = true;
            dfsCombinationSum2(candidates, target - candidates[i], res, path, i + 1, used);
            used[i] = false;
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        dfsCombinationSum(candidates, target, res, path, 0);
        return res;
    }

    public void dfsCombinationSum(int[] candidates, int target, List<List<Integer>> res, List<Integer> path, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        dfsCombinationSum(candidates, target, res, path, idx + 1);
        if (target - candidates[idx] >= 0) {
            path.add(candidates[idx]);
            dfsCombinationSum(candidates, target - candidates[idx], res, path, idx);
            path.remove(path.size() - 1);
        }
    }

    public boolean isHappy(int n) {
        int slow = n;
        int fast = getNext(n);
        while (fast != 1 && slow != fast) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;
    }

    private int getNext(int n) {
        int total = 0;
        while (n > 0) {
            int d = n % 10;
            n /= 10;
            total += d * d;
        }
        return total;
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        boolean[] visit = new boolean[nums.length];
        dfsPermuteUnique(nums, res, path, visit);
        return res;
    }

    private void dfsPermuteUnique(int[] nums, List<List<Integer>> res, List<Integer> path, boolean[] visit) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
        }
        for (int i = 0; i < nums.length; i++) {
            if (visit[i] || (i > 0 && nums[i] == nums[i - 1] && !visit[i - 1])) {
                continue;
            }
            path.add(nums[i]);
            visit[i] = true;
            dfsPermuteUnique(nums, res, path, visit);
            visit[i] = false;
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfsSubsetsWithDup(nums, res, path, 0);
        return res;
    }

    private void dfsSubsetsWithDup(int[] nums, List<List<Integer>> res, List<Integer> path, int start) {
        res.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            path.add(nums[i]);
            dfsSubsetsWithDup(nums, res, path, i + 1);
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int n = 1 << nums.length;
        for (int i = 0; i < n; i++) {
            List<Integer> cur = new ArrayList<>();
            for (int j = 0; j < nums.length; j++) {
                if (((1 << j) & i) != 0) {
                    cur.add(nums[j]);
                }
            }
            res.add(cur);
        }
        return res;
    }

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        path.add(0);
        dfsPathsSourceTarget(graph, path, res, 0);
        return res;
    }

    private void dfsPathsSourceTarget(int[][] graph, List<Integer> path, List<List<Integer>> res, int cur) {
        if (cur == graph.length - 1) {
            res.add(new ArrayList<>(path));
            return;
        }
        int[] ints = graph[cur];
        for (int anInt : ints) {
            path.add(anInt);
            dfsPathsSourceTarget(graph, path, res, anInt);
            path.remove(path.size() - 1);
        }
    }

    public void solve(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        if (board.length == 1) {
            return;
        }
        for (int i = 0; i < m; i++) {
            dfsSolve(board, i, 0);
            dfsSolve(board, i, n - 1);
        }
        for (int i = 1; i < n - 1; i++) {
            dfsSolve(board, 0, i);
            dfsSolve(board, m - 1, i);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'S') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void dfsSolve(char[][] grid, int x, int y) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || grid[x][y] != 'O') {
            return;
        }
        grid[x][y] = 'S';
        dfsSolve(grid, x - 1, y);
        dfsSolve(grid, x + 1, y);
        dfsSolve(grid, x, y - 1);
        dfsSolve(grid, x, y + 1);
    }

    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid[0][0] == 1) {
            return -1;
        }
        int m = grid.length;
        int n = grid[0].length;
        Deque<int[]> queue = new LinkedList<>();
        queue.add(new int[]{0, 0});
        grid[0][0] = 1;
        int[][] d = new int[][]{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        int path = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cell = queue.pop();
                int x = cell[0];
                int y = cell[1];
                if (x == m - 1 && y == n - 1) {
                    return path;
                }
                for (int[] ints : d) {
                    int mx = x + ints[0];
                    int my = y + ints[1];
                    if (mx >= 0 && mx < m && my >= 0 && my < n && grid[mx][my] == 0) {
                        queue.offer(new int[]{mx, my});
                        grid[mx][my] = 1;
                    }
                }
            }
            path++;
        }
        return -1;
    }

    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) {
            return false;
        }
        return checkSubtree(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    private boolean checkSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) {
            return true;
        }
        if (root == null || subRoot == null || root.val != subRoot.val) {
            return false;
        }
        return checkSubtree(root.left, subRoot.left) && checkSubtree(root.right, subRoot.right);
    }

    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Deque<Node> nodes = new LinkedList<>();
        nodes.offer(root);
        while (!nodes.isEmpty()) {
            int size = nodes.size();
            for (int i = 0; i < size; i++) {
                var node = nodes.poll();
                if (i != size - 1) {
                    node.next = nodes.peek();
                }
                if (node.left != null) {
                    nodes.offer(node.left);
                }
                if (node.right != null) {
                    nodes.offer(node.right);
                }
            }
        }
        return root;
    }

    public int findCircleNum(int[][] isConnected) {
        int sum = 0;
        int n = isConnected.length;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                sum++;
                dfsFindCircleNum(isConnected, i, visited);
            }
        }
        return sum;
    }

    private void dfsFindCircleNum(int[][] isConnected, int city, boolean[] visited) {
        for (int i = 0; i < isConnected.length; i++) {
            if (isConnected[city][i] == 1 && !visited[i]) {
                visited[i] = true;
                dfsFindCircleNum(isConnected, i, visited);
            }
        }
    }

    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    dfsNumIslands(grid, i, j);
                    sum++;
                }
            }
        }
        return sum;
    }

    private void dfsNumIslands(char[][] grid, int x, int y) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || grid[x][y] == '0') {
            return;
        }
        grid[x][y] = '0';
        dfsNumIslands(grid, x - 1, y);
        dfsNumIslands(grid, x + 1, y);
        dfsNumIslands(grid, x, y - 1);
        dfsNumIslands(grid, x, y + 1);
    }

    public int minSubArrayLen(int target, int[] nums) {
        int left = 0, right = 0;
        int res = Integer.MAX_VALUE;
        int sum = 0;
        while (right < nums.length) {
            sum += nums[right];
            while (sum >= target) {
                res = Math.min(res, right - left + 1);
                sum -= nums[left];
                left++;
            }
            right++;
        }
        if (res == Integer.MAX_VALUE) {
            return 0;
        }
        return res;
    }

    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) {
            return 0;
        }
        int res = 0;
        int product = 1;
        int left = 0, right = 0;
        while (right < nums.length) {
            product *= nums[right];
            while (product >= k) {
                product /= nums[left];
                left++;
            }
            res += right - left + 1;
            right++;
        }
        return res;
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int n = p.length();
        if (p.length() > s.length()) {
            return res;
        }
        int[] cacheS = new int[26];
        int[] cacheP = new int[26];
        for (int i = 0; i < n; i++) {
            cacheS[s.charAt(i) - 'a']++;
            cacheP[p.charAt(i) - 'a']++;
        }

        if (Arrays.equals(cacheS, cacheP)) {
            res.add(0);
        }

        for (int i = 0; i < s.length() - n; i++) {
            cacheS[s.charAt(i) - 'a']--;
            cacheS[s.charAt(i + n) - 'a']++;
            if (Arrays.equals(cacheS, cacheP)) {
                res.add(i + 1);
            }
        }
        return res;
    }

    public int maxArea(int[] height) {
        int max = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int h = Math.min(height[left], height[right]);
            int cur = (right - left) * h;
            max = Math.max(cur, max);
            if (height[left] <= height[right]) {
                int nextLeft = left + 1;
                while (nextLeft < height.length && height[nextLeft] < height[left]) {
                    nextLeft++;
                }
                left = nextLeft;
            } else {
                int nextRight = right - 1;
                while (nextRight >= 0 && height[nextRight] < height[right]) {
                    nextRight--;
                }
                right = nextRight;
            }
        }
        return max;
    }

    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> res = new ArrayList<>();
        int firstIndex = 0, secondIndex = 0;
        while (firstIndex < firstList.length && secondIndex < secondList.length) {
            int leftMax = Math.max(firstList[firstIndex][0], secondList[secondIndex][0]);
            int rightMin = Math.min(firstList[firstIndex][1], secondList[secondIndex][1]);
            if (leftMax <= rightMin) {
                res.add(new int[]{leftMax, rightMin});
            }
            if (firstList[firstIndex][1] > secondList[secondIndex][1]) {
                secondIndex++;
            } else if (firstList[firstIndex][1] < secondList[secondIndex][1]) {
                firstIndex++;
            } else {
                firstIndex++;
                secondIndex++;
            }
        }
        return res.toArray(new int[0][]);
    }

    public boolean backspaceCompareDoublePoints(String s, String t) {
        return backspaceStr(s).equals(backspaceStr(t));
    }

    public String backspaceStr(String s) {
        char[] charS = s.toCharArray();
        int k = 0;
        for (char ch : charS) {
            if (ch == '#') {
                if (k != 0) {
                    k--;
                }
            } else {
                charS[k++] = ch;
            }
        }
        return new String(charS, 0, k);
    }

    public boolean backspaceCompare(String s, String t) {
        Deque<Character> stack1 = backspace(s);
        Deque<Character> stack2 = backspace(t);
        if (stack1.size() != stack2.size()) {
            return false;
        }
        while (!stack1.isEmpty()) {
            if (!stack1.pop().equals(stack2.pop())) {
                return false;
            }
        }
        return true;
    }

    private Deque<Character> backspace(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '#') {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(c);
            }
        }
        return stack;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int pre1 = Integer.MAX_VALUE, pre2 = Integer.MAX_VALUE, pre3 = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            int cur = nums[i];
            if (pre1 == cur) {
                continue;
            }
            pre1 = cur;
            int sum1 = -cur;
            pre2 = Integer.MAX_VALUE;
            for (int j = i + 1; j < nums.length; j++) {
                int cur2 = nums[j];
                if (pre2 == cur2) {
                    continue;
                }
                pre2 = cur2;
                int sum2 = sum1 - nums[j];
                for (int k = j + 1; k < nums.length; k++) {
                    if (nums[k] == sum2) {
                        List<Integer> l = new ArrayList<>();
                        l.add(cur);
                        l.add(cur2);
                        l.add(nums[k]);
                        res.add(l);
                        break;
                    }
                }
            }
        }
        return res;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode tmp = res;
        while (tmp.next != null && tmp.next.next != null) {
            if (tmp.next.val == tmp.next.next.val) {
                int val = tmp.next.val;
                while (tmp.next != null && tmp.next.val == val) {
                    tmp.next = tmp.next.next;
                }
            } else {
                tmp = tmp.next;
            }
        }
        return res.next;
    }

    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int rl = 0, rh = m - 1;
        while (rl <= rh) {
            int mid = (rl + rh) >>> 1;
            if (matrix[mid][0] == target) {
                return true;
            } else if (matrix[mid][0] > target) {
                rh = mid - 1;
            } else {
                rl = mid + 1;
            }
        }
        int row = rh;
        if (row < 0) {
            return false;
        }
        int cl = 0, cr = n - 1;
        while (cl <= cr) {
            int mid = (cl + cr) >>> 1;
            int cur = matrix[row][mid];
            if (cur == target) {
                return true;
            } else if (cur > target) {
                cr = mid - 1;
            } else {
                cl = mid + 1;
            }
        }
        return false;
    }

    public int mySqrt(int x) {
        if (x == 0 || x == 1) {
            return x;
        }
        int left = 0, right = x;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            long cur = (long) mid * mid;
            if (cur == x) {
                return mid;
            } else if (cur > x) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left - 1;
    }

    public int missingNumber(int[] nums) {
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] == mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right + 1;
    }

    public boolean halvesAreAlike(String s) {
        List<Character> characters = List.of('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U');
        int n = s.length();
        int sum = 0;
        for (int i = 0; i < n / 2; i++) {
            sum += characters.contains(s.charAt(i)) ? 1 : 0;
            sum -= characters.contains(s.charAt(i + n / 2)) ? 1 : 0;
        }
        return sum == 0;
    }

    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] >= nums[0]) {
                if (nums[mid] > target && target >= nums[0]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[nums.length - 1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int left = binarySearch(nums, target);
        int right = binarySearch(nums, target + 1) - 1;
        if (left <= right && left >= 0 && right <= nums.length - 1) {
            return new int[]{left, right};
        }
        return new int[]{-1, -1};
    }

    public int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right + 1) >>> 1;
            if (nums[mid] >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    public int maxRepeating(String sequence, String word) {
        int left = 0;
        int res = 0;
        int cur = 0;
        while (left <= sequence.length() - word.length()) {
            boolean flag = true;
            for (int i = left; i < left + word.length(); i++) {
                if (sequence.charAt(i) != word.charAt(i - left)) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                left = left + word.length();
                cur++;
            } else {
                res = Math.max(cur, res);
                left = left - cur * word.length() + 1;
                cur = 0;
            }
        }
        return Math.max(cur, res);
    }

    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int index1 = 0;
        int index2 = 0;
        while (index1 < word1.length() && index2 < word2.length()) {
            sb.append(word1.charAt(index1));
            sb.append(word2.charAt(index2));
            index1++;
            index2++;
        }
        while (index1 < word1.length()) {
            sb.append(word1.charAt(index1));
            index1++;
        }
        while (index2 < word2.length()) {
            sb.append(word2.charAt(index2));
            index2++;
        }
        return sb.toString();
    }

    public int minCameraCover(TreeNode root) {
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        int level = 1;
        int cnt = 0;
        while (!deque.isEmpty()) {
            boolean flag = level % 2 == 0;
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.poll();

            }
            level++;
        }
        return 1;
    }
}
