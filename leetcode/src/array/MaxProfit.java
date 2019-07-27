package array;

public class MaxProfit {
    public static void main(String[] args) {
        int[] prices = new int[]{7,1,5,3,6,4};
        System.out.println(new MaxProfit().maxProfit(prices));
    }

    public int maxProfit(int[] prices) {
        int maxValue = 0;
        for (int i = 0; i < prices.length-1; i++) {
            if(prices[i+1]>prices[i]){
                maxValue += prices[i+1] - prices[i];
            }
        }
        return maxValue;
    }
}
