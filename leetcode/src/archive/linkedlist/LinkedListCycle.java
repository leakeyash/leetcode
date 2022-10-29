package archive.linkedlist;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class LinkedListCycle {
    public static void main(String[] args) {
        ListNode listNode1 = new ListNode(1);
        ListNode listNode2 = new ListNode(1);
        List<ListNode> listNodes = Arrays.asList(listNode1, listNode2);
        System.out.println(listNodes.indexOf(listNode2));

        Set<ListNode> listNodeSet = new HashSet<>();
        listNodeSet.add(listNode1);
        listNodeSet.add(listNode2);
        System.out.println(listNodeSet);
    }
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode first = head;
        ListNode second = head.next;
        while(second != first) {
            if (second == null || second.next == null) {
                return false;
            }
            // first 往后移一位， second 往后移两位， 最后如果得到的对象相同则组成环
            first = first.next;
            second = second.next.next;
        }
        return true;
    }

    public boolean hasCycle1(ListNode head) {
        Set<ListNode> listNodeSet = new HashSet<>();
        ListNode curr = head;
        while(curr != null){
            if(listNodeSet.contains(curr)){
                return true;
            }
            listNodeSet.add(curr);
            curr = curr.next;
        }
        return false;
    }
}
