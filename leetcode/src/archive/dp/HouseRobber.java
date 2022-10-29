package archive.dp;

public class HouseRobber {
    public int rob(int[] nums) {
        int l = nums.length;
        if(l == 0){
            return 0;
        }
        int[] dp = new int[l+1];
        dp[0] = 0;
        dp[1] = nums[0];
        for(int i= 2; i<=l;i++){
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i-1]);
        }
        return dp[l];
    }
}
