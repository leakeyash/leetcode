package string;

public class ReverseNumber {
    public static void main(String[] args) {
        System.out.println(new ReverseNumber().reverse(12345));
    }
    public int reverse(int x) {
        char[] chars = String.valueOf(x).toCharArray();
        int start = 0;
        if(chars[0]=='-'){
            start = 1;
        }
        for(int i=start;i<(chars.length)/2 + start;i++){
            char temp = chars[i];
            int destIndex = chars.length - 1 -i + start;
            chars[i] = chars[destIndex];
            chars[destIndex] = temp;
        }
        System.out.println(chars);
        return 0;
    }
}
