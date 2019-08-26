package dp;

public class BestTimetoBuyandSellStock {
    public int maxProfit(int[] prices) {
        int l = prices.length;
        if(l == 0){
            return 0;
        }
        int[] dp = new int[l-1];
        for(int i =1; i< l ;i++){
            dp[i-1] = prices[i] - prices[i-1];
        }
        int temp = 0;
        int max = 0;
        for(int i: dp){
            if(temp <= 0){
                temp = i;
            } else {
                temp += i;
            }
            max = Math.max(max, temp);
        }
        return Math.max(max, 0);
    }
}
