package linkedlist;

public class PalindromeLinkedList {
    public static void main(String[] args) {
        new PalindromeLinkedList().isPalindrome(ListNodeFactory.newListNode(1,1,2,1));
    }
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null)
            return true;
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode iter = slow;
        ListNode pre = null;
        while (iter != null) {
            ListNode temp = iter.next;
            iter.next = pre;
            pre = iter;
            iter = temp;
        }
        iter = head;
        ListNode tail = pre;
        while (tail != null) {
            if (iter.val != tail.val) {
                return false;
            }
            iter = iter.next;
            tail = tail.next;
        }
        return true;
    }
}
