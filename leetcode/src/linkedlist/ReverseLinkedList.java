package linkedlist;

public class ReverseLinkedList {
    public static void main(String[] args) {
        ListNode result = new ReverseLinkedList().reverseList(ListNodeFactory.newListNode(1,2,3,4,5));
        ListNodeFactory.printListNode(result);
    }
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while(curr != null){
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

}
