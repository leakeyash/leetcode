package linkedlist;

public class ListNodeFactory {

    public static ListNode newListNode(int... values){
        ListNode head = new ListNode(values[0]);
        int index = 1;
        ListNode point = head;
        while(index < values.length){
            point.next = new ListNode(values[index]);
            point = point.next;
            index ++;
        }
        printListNode(head);
        return head;
    }

    public static void printListNode(ListNode listNode){
        while(listNode!=null){
            System.out.print(listNode.val);
            listNode = listNode.next;
            if(listNode != null){
                System.out.print("->");
            }
        }
        System.out.println();
    }
}
