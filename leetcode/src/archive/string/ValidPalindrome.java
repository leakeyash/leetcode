package archive.string;

public class ValidPalindrome {
    public static void main(String[] args) {
        System.out.println(new ValidPalindrome().isPalindrome("A man, a plan, a canal: Panama"));
        System.out.println(new ValidPalindrome().isPalindrome("race a car"));
        System.out.println(new ValidPalindrome().isPalindrome("0-P0,"));
    }
    public boolean isPalindrome(String s) {
        char[] chars = s.toLowerCase().toCharArray();
        int i = 0;
        int j = chars.length-1;
        while(i <= j){

            while( !Character.isLetterOrDigit(chars[i]) && i < j){
                i++;
            }
            while( !Character.isLetterOrDigit(chars[j]) && i < j){
                j--;
            }
            if(chars[i] == chars[j]){
                i++;
                j--;
            } else{
                return false;
            }
        }
        return true;
    }
}
