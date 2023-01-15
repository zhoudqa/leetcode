import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

public abstract class SolutionBase {

    static class TreeNode {

        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    static class ListNode {

        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    static class Node {

        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public static ListNode stringToListNode(String s) {
        final List<Integer> ints = Arrays.stream(s.substring(1, s.length() - 1).split(",")).map(Integer::parseInt)
                .collect(Collectors.toList());
        ListNode head = new ListNode(ints.get(0));
        ListNode pre = head;
        ListNode node;
        for (int i = 1; i < ints.size(); i++) {
            node = new ListNode(ints.get(i));
            pre.next = node;
            pre = node;
        }
        return head;
    }

    public static int[] stringToArray(String s) {
        String[] split = s.substring(1, s.length() - 1).split(",");
        int[] res = new int[split.length];
        for (int i = 0; i < split.length; i++) {
            res[i] = Integer.parseInt(split[i]);
        }
        return res;
    }

    public static int[][] stringToMatrix(String s) {
        final String[] elements = s.substring(2, s.length() - 2).split("],\\[");
        final int h = elements.length;
        final int w = elements[0].split(",").length;
        int[][] res = new int[h][w];
        for (int i = 0; i < elements.length; i++) {
            final String[] split = elements[i].split(",");
            for (int j = 0; j < split.length; j++) {
                res[i][j] = Integer.parseInt(split[j]);
            }
        }
        for (final int[] re : res) {
            for (final int i : re) {
                System.out.print(i + " ");
            }
            System.out.println();
        }
        return res;
    }

    public static TreeNode bfsBuild(String s) {
        final int[] ints = stringToArray(s);
        TreeNode root = new TreeNode(ints[0]);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while (i < ints.length) {
            final TreeNode poll = queue.poll();
            poll.left = new TreeNode(ints[i++]);
            queue.add(poll.left);
            if (i < ints.length) {
                poll.right = new TreeNode(ints[i++]);
                queue.add(poll.right);
            } else {
                break;
            }
        }
        return root;
    }


}
