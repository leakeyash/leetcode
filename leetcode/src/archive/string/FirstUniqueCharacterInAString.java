package archive.string;

import java.util.HashMap;
import java.util.Map;

public class FirstUniqueCharacterInAString {
    public static void main(String[] args) {
        System.out.println(new FirstUniqueCharacterInAString().firstUniqChar("lovveleettccodde"));
        System.out.println(new FirstUniqueCharacterInAString().firstUniqChar2("lovveleettccodde"));
    }
    public int firstUniqChar(String s) {
        HashMap<Character,Integer> map = new HashMap<>(s.length());
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if(!map.containsKey(chars[i])){
                map.put(chars[i],i);
            }else{
                map.put(chars[i],-1);
            }
        }
        int result = chars.length;
        for(Map.Entry<Character,Integer> entry: map.entrySet()){
            if(entry.getValue() != -1){
                result = Math.min(entry.getValue(),result);
            }
        }
        if(result==chars.length){
            return -1;
        }
        return result;
    }

    public int firstUniqChar2(String s) {
        int index = -1;
        for (char ch = 'a'; ch <= 'z'; ch++) {
            int startIndex = s.indexOf(ch);
            if (startIndex != -1 && startIndex == s.lastIndexOf(ch)) {
                index = (index == -1 || index > startIndex) ? startIndex : index;
            }
        }
        return index;
    }
}
