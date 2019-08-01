package linkedlist;

public class RemoveNthNodeFromEndOfList {
    public static void main(String[] args) {
        ListNode result = new RemoveNthNodeFromEndOfList().removeNthFromEnd(ListNodeFactory.newListNode(1,2,3,4,5),2);
        ListNodeFactory.printListNode(result);
        result = new RemoveNthNodeFromEndOfList().removeNthFromEnd(ListNodeFactory.newListNode(1),1);
        ListNodeFactory.printListNode(result);
        result = new RemoveNthNodeFromEndOfList().removeNthFromEnd(ListNodeFactory.newListNode(1,2),1);
        ListNodeFactory.printListNode(result);
        result = new RemoveNthNodeFromEndOfList().removeNthFromEnd(ListNodeFactory.newListNode(1,2),2);
        ListNodeFactory.printListNode(result);
    }
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int count = getNumber(head, n + 1);
        if(count == n){
            head = head.next;
        }
        return head;
    }

    public int getNumber(ListNode head, int n){
        if(head.next == null)
        {
            return 1;
        }
        int count = getNumber(head.next, n) + 1;
        if(count == n){
            head.next = head.next == null ? null : head.next.next;
        }
        return count;
    }

    public ListNode removeNthFromEnd1(ListNode head, int n) {
        //定义2个指针
        ListNode x = head;
        ListNode y = head;

        int i = n;

        //往后移n位
        while (i > 0) {
            if (y.next == null) {
                return head.next;
            }
            y = y.next;
            i--;
        }

        //两个指针同时往后移，y到底的时候，x正好是倒数第n+1个
        while (y.next != null) {
            y = y.next;
            x = x.next;
        }

        x.next = x.next.next;

        return head;
    }

}
