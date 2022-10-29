package archive.string;

public class CountAndSay {
    public static void main(String[] args) {
        String result = new CountAndSay().countAndSay(1);
        System.out.println(result);
        assert "1".equals(result);

        result = new CountAndSay().countAndSay(4);
        System.out.println(result);
        assert "1211".equals(result);
    }

    public String countAndSay(int n) {
        if(n == 1){
            return "1";
        }
        String curr = "1";
        int count = 2;
        while(count <= n){
            int i = 0;
            StringBuilder temp = new StringBuilder();
            while(i < curr.length()){
                int value = curr.charAt(i) - '0';
                int j = i+1;
                while(j<curr.length() && curr.charAt(j) - '0'==value){
                    j++;
                }
                temp.append(j - i).append(value);
                i = j;
            }
            curr = temp.toString();
            count ++ ;
        }
        return curr;
    }
}
