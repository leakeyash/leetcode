package dp;

public class MaximumSubarray {
    public int maxSubArray(int[] nums) {
        int l = nums.length;
        int temp = 0;
        int max = Integer.MIN_VALUE;
        for(int i=0; i< l ;i++){
            if(temp <= 0){
                temp = nums[i];
            } else {
                temp += nums[i];
            }
            max = Math.max(max, temp);
        }
        return max;
    }
}
