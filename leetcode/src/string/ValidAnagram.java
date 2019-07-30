package string;

import java.util.HashMap;

public class ValidAnagram {
    public static void main(String[] args) {
        System.out.println(new ValidAnagram().isAnagram("anagram", "nagaram"));
        System.out.println(new ValidAnagram().isAnagram("", ""));
    }

    public boolean isAnagram(String s, String t) {
        if (s == null || t == null || s.length() != t.length()) {
            return false;
        }
        if (s.length() == 0) {
            return true;
        }
        char[] ss = s.toCharArray();
        char[] ts = t.toCharArray();
        HashMap<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < ss.length; i++) {
            char value = ss[i];
            if(map.containsKey(value)){
                map.put(value, map.get(value)+1);
            } else{
                map.put(value, 1);
            }
        }
        for (int i = 0; i < ts.length; i++) {
            char value = ts[i];
            if(map.containsKey(value)){
                map.put(value, map.get(value)-1);
            } else{
                return false;
            }
        }
        for(Integer item: map.values()){
            if(item != 0){
                return false;
            }
        }
        return true;
    }

    public boolean isAnagram2(String s, String t) {
        int[] arr=new int[128];
        for(char m:s.toCharArray()) {
            arr[m]+=1;
        }
        for(char m:t.toCharArray()) {
            arr[m]-=1;
        }
        for(int i='a';i<='z';i++) {
            if(arr[i]!=0) {
                return false;
            }
        }
        return true;
    }
}
