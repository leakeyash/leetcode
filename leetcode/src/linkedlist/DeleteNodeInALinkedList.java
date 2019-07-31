package linkedlist;

public class DeleteNodeInALinkedList {
    public static void main(String[] args) {
        ListNodeFactory.newListNode(1,2,3);
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
