import java.util.*;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

public class Solution1 {


    public int numDifferentIntegers(String word) {
        Set<String> integers = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (final char c : word.toCharArray()) {
            if (c >= 48 && c <= 57) {
                sb.append(c);
            } else if (sb.length() != 0) {
                integers.add(parseNumber(sb.toString()));
                sb = new StringBuilder();
            }
        }
        if (sb.length() != 0) {
            integers.add(parseNumber(sb.toString()));
        }
        return integers.size();
    }

    private String parseNumber(String str) {
        int index = 0;
        while (index < str.length() && str.charAt(index) == 48) {
            index++;
        }
        return index == str.length() ? "0" : str.substring(index);
    }

    //给你一个n x n矩阵matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
    //请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
    //你必须找到一个内存复杂度优于O(n2) 的解决方案。
    //n == matrix.length
    //n == matrix[i].length
    //1 <= n <= 300
    //-109 <= matrix[i][j] <= 109
    //题目数据 保证 matrix 中的所有行和列都按 非递减顺序 排列
    //1 <= k <= n^2
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];
        //找到left到right之间的数，这个数为第N小，如果N<=k,则向左继续二分，否则向右继续二分
        while (left < right) {
            //中间数，不一定在数组中
            int mid = left + (right - left) / 2;
            if (inLeft(matrix, k, n, mid)) {
                //mid指针往左
                right = mid;
            } else {
                //mid指针往右
                left = mid + 1;
            }
        }
        return left;

    }

    private boolean inLeft(int[][] matrix, int k, int n, int mid) {
        //从左下角开始遍历
        int i = n - 1;
        int j = 0;
        int smaller = 0;
        while (i >= 0 && j <= n - 1) {
            if (matrix[i][j] <= mid) {
                //比mid小的在他上面，个数要+1，然后往右移动
                smaller += i + 1;
                j++;
            } else {
                //减小matrix[i][j]看看
                i--;
            }
        }
        return smaller >= k;
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

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        } else if (lists.length == 1) {
            return lists[0];
        } else {
            return mergeLists(lists, 0, lists.length - 1);

        }

    }

    //二分合并
    private ListNode mergeLists(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        int mid = left + (right - left) / 2;
        ListNode leftNode = mergeLists(lists, left, mid);
        ListNode rightNode = mergeLists(lists, mid + 1, right);

        return mergeListNodes(leftNode, rightNode);
    }

    private ListNode mergeListNodes(ListNode list, ListNode list1) {
        if (list == null || list1 == null) {
            return list != null ? list : list1;
        }
        ListNode head = new ListNode(0);
        ListNode tail = head;
        ListNode left = list;
        ListNode right = list1;
        while (left != null && right != null) {
            if (left.val < right.val) {
                //复用以节省空间
                tail.next = left;
                left = left.next;
            } else {
                tail.next = right;
                right = right.next;
            }
            tail = tail.next;
        }
        tail.next = left == null ? right : left;
        return head.next;
    }

    //优先队列(堆)方式
    public ListNode mergeKListsWithHeap(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.val));
        pq.addAll(Arrays.asList(lists).stream().filter(listNode -> listNode != null).collect(Collectors.toList()));
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (!pq.isEmpty()) {
            tail.next = pq.poll();
            tail = tail.next;
            //剩下的一截入队列
            if (tail.next != null) {
                pq.add(tail.next);
            }
        }
        return head.next;
    }

    //给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }
        //维护大小为k的最小堆
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        frequencyMap.forEach((val, frequency) -> {
            //频率比最小的还低，就不需要入堆了，保证堆的大小为k
            if (pq.size() == k) {
                if (pq.peek()[1] < frequency) {
                    pq.poll();
                    pq.add(new int[]{val, frequency});
                }
            } else {
                pq.add(new int[]{val, frequency});
            }
        });
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll()[0];
        }
        return res;
    }

    //给你一个整数数组 nums，有一个大小为k的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k个数字。滑动窗口每次只向右移动一位。
    //
    //返回 滑动窗口中的最大值 。
    public int[] maxSlidingWindow(int[] nums, int k) {
        int resLen = nums.length - (k - 1);
        int[] res = new int[resLen];
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> o2[1] - o1[1]);
        for (int i = 0; i < k; i++) {
            pq.add(new int[]{i, nums[i]});
        }
        int idx = 0;
        res[idx++] = pq.peek()[1];
        for (int i = k; i < nums.length; i++) {
            pq.add(new int[]{i, nums[i]});
            int pre = i - k;
            //低效移除O(N) pq.removeIf(o -> o[0] == pre);
            while (pq.peek()[0] <= pre) {
                //高效移除去掉错误答案 i-k坐标的元素
                pq.poll();
            }
            res[idx++] = pq.peek()[1];
        }
        return res;

    }


    public static void main(String[] args) {
        final Solution1 solution = new Solution1();
//        solution.numDifferentIntegers("0a0");
//        solution.kthSmallest(stringToMatrix("[[1,3,5],[6,7,12],[11,14,14]]"), 3);
//        solution.topKFrequent(stringToArray("[1,1,1,2,2,3]"), 2);
        solution.maxSlidingWindow(stringToArray("[1,-1]"), 1);
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
        return res;
    }

}
