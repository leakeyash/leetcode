import helper.ListNode;
import helper.Node;
import helper.TreeNode;

import java.util.*;
import java.util.logging.Level;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Contest {
    public static void main(String[] args) {
        System.out.println("Hello");
        new Contest().makeIntegerBeautiful(8L, 2);
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
