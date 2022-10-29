package archive.string;

public class LongestCommonPrefix {
    public static void main(String[] args) {
        String result = new LongestCommonPrefix().longestCommonPrefix(new String[]{"flower","flow","flight"});
        System.out.println(result);
        assert "fl".equals(result);

        result = new LongestCommonPrefix().longestCommonPrefix(new String[]{"dog","racecar","car"});;
        System.out.println(result);
        assert "".equals(result);
    }

    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0){
            return "";
        }
        if(strs.length == 1){
            return strs[0];
        }
        String common = strs[0];
        int commonLength = common.length();
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            int length = strs[i].length();
            while(j < commonLength && j < length){
                if(common.charAt(j) == strs[i].charAt(j)){
                    j++;
                } else{
                    break;
                }
            }
            if(j == 0){
                return "";
            } else if (j< commonLength){
                common = common.substring(0,j);
                commonLength = j;
            }
        }
        return common;
    }
}
