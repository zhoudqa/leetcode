import java.util.*;
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
        pq.addAll(Arrays.stream(lists).filter(Objects::nonNull).collect(Collectors.toList()));
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
                //高效移除去掉错误答案 i-k坐标的元素，因为只要peek的数字坐标在窗口中就可以
                pq.poll();
            }
            res[idx++] = pq.peek()[1];
        }
        return res;

    }

    static class TreeNode {

        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    static class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) {
                return "X";
            }
            String left = "(" + serialize(root.left) + ")";
            String right = "(" + serialize(root.right) + ")";
            return left + root.val + right;
        }

        public TreeNode deserialize(String data) {
            int[] ptr = {0};
            return parse(data, ptr);
        }

        public TreeNode parse(String data, int[] ptr) {
            if (data.charAt(ptr[0]) == 'X') {
                ++ptr[0];
                return null;
            }
            TreeNode cur = new TreeNode(0);
            cur.left = parseSubtree(data, ptr);
            cur.val = parseInt(data, ptr);
            cur.right = parseSubtree(data, ptr);
            return cur;
        }

        public TreeNode parseSubtree(String data, int[] ptr) {
            ++ptr[0]; // 跳过左括号
            TreeNode subtree = parse(data, ptr);
            ++ptr[0]; // 跳过右括号
            return subtree;
        }

        public int parseInt(String data, int[] ptr) {
            int x = 0, sgn = 1;
            if (!Character.isDigit(data.charAt(ptr[0]))) {
                sgn = -1;
                ++ptr[0];
            }
            while (Character.isDigit(data.charAt(ptr[0]))) {
                x = x * 10 + data.charAt(ptr[0]++) - '0';
            }
            return x * sgn;
        }

    }

    //给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
    public int longestSubstring(String s, int k) {
        return longestSubstring(s, 0, s.length() - 1, k);
    }


    //s的[left,right]子串中最长的每个字符出现次数都不小于k的子串
    private int longestSubstring(String s, int left, int right, int k) {
        int[] count = new int[26];
        for (int i = left; i <= right; i++) {
            count[s.charAt(i) - 'a']++;
        }
        int split = -1;
        for (int i = 0; i < 26; i++) {
            if (count[i] > 0 && count[i] < k) {
                //找到第一个不满足k个条件的char
                split = i;
            }
        }
        //找不到不满足条件的为退出条件
        if (split == -1) {
            return right - left + 1;
        }
        int i = left;
        int ret = Integer.MIN_VALUE;
        //从left到right比较每一个使用split分隔的子串
        while (i <= right) {
            while (i <= right && s.charAt(i) - 'a' == split) {
                i++;
            }
            //第一个不为split的index
            int newLeft = i;
            while (i <= right && s.charAt(i) - 'a' != split) {
                i++;
            }
            //第一个split的index
            int newRight = i - 1;
            ret = Math.max(ret, longestSubstring(s, newLeft, newRight, k));
        }
        return ret;
    }

    //给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
    public int missingNumber(int[] nums) {
        int sum = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
        }
        return n * (n + 1) / 2 - sum;
    }

    //给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    //请你设计并实现时间复杂度为O(n) 的算法解决此问题。
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int ret = 0;
        for (int num : set) {
            //避免无用的循环
            if (!set.contains(num - 1)) {
                int j = 0;
                while (set.contains(num + (++j))) {
                }
                ret = Math.max(ret, j);
            }
        }
        return ret;

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

    Map<Node, Node> cache = new HashMap<>();

    //给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
    //构造这个链表的 深拷贝。
    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        if (!cache.containsKey(head)) {
            Node copy = new Node(head.val);
            cache.put(head, copy);
            copy.next = copyRandomList(head.next);
            copy.random = copyRandomList(head.random);
        }
        return cache.get(head);
    }

    //给你一个链表的头节点 head ，判断链表中是否有环。
    public boolean hasCycle(ListNode head) {
        Set<ListNode> cache = new HashSet<>();
        ListNode node = head;
        while (node != null) {
            if (cache.contains(node)) {
                return true;
            }
            cache.add(node);
            node = node.next;
        }
        return false;
    }

    //快慢指针
    public boolean hasCycleBest(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    //给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表
    public ListNode sortList(ListNode head) {
        List<Integer> array = new ArrayList<>();
        ListNode node = head;
        while (node != null) {
            array.add(node.val);
            node = node.next;
        }
        if (array.isEmpty()) {
            return null;
        }
        array.sort(Integer::compareTo);
        ListNode newHead = new ListNode(array.get(0));
        ListNode node1 = newHead;
        for (int i = 1; i < array.size(); i++) {
            node1.next = new ListNode(array.get(i));
            node1 = node1.next;
        }
        return newHead;
    }

    public ListNode sortListBest(ListNode head) {
        if (head == null) {
            return null;
        }
        int len = 0;
        ListNode node = head;
        while (node != null) {
            len++;
            node = node.next;
        }
        ListNode preHead = new ListNode(-1);
        preHead.next = head;
        for (int duan = 1; duan < len; duan *= 2) {
            //每次循环之后，内部总是段段有序的
            ListNode pre = preHead, cur = preHead.next;
            while (cur != null) {
                //pre指向前一个段的末尾，cur指向当前段的头
                ListNode head1 = cur;
                for (int i = 1; i < duan && cur.next != null; i++) {
                    cur = cur.next;
                }
                ListNode head2 = cur.next;
                //断开Merge两段
                cur.next = null;
                cur = head2;
                for (int i = 1; i < duan && cur != null && cur.next != null; i++) {
                    cur = cur.next;
                }
                ListNode next = null;
                if (cur != null) {
                    next = cur.next;
                    //断开未排序节点的下段连接
                    cur.next = null;
                }
                //用于连接前后分段
                pre.next = merge2ListNode(head1, head2);
                while (pre.next != null) {
                    pre = pre.next;
                }
                cur = next;
            }
        }
        return preHead.next;

    }

    public ListNode merge2ListNode(ListNode head1, ListNode head2) {
        ListNode preHead = new ListNode();
        ListNode temp = preHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val < temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        }
        if (temp2 != null) {
            temp.next = temp2;
        }
        return preHead.next;
    }

    //给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tail = head.next;
        ListNode nextHead = reverseList(head.next);
        head.next = null;
        tail.next = head;
        return nextHead;
    }

    //删除node节点
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
    public String largestNumber(int[] nums) {
        String s = Arrays.stream(nums).mapToObj(String::valueOf).sorted((s1, s2) -> (s2 + s1).compareTo(s1 + s2))
                .reduce((a, b) -> a + b).orElseThrow(RuntimeException::new);
        return '0' == s.charAt(0) ? "0" : s;
    }

    int maxPathSum = Integer.MIN_VALUE;

    //路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
    //
    //路径和 是路径中各节点值的总和。
    //
    //给你一个二叉树的根节点 root ，返回其 最大路径和 。
    //递归
    public int maxPathSum(TreeNode root) {
        maxSingleSide(root);
        return maxPathSum;
    }

    //该函数返回包含root节点的最大路径和
    private int maxSingleSide(TreeNode root) {
        if (root == null) {
            return 0;
        }
        final int leftMax = Math.max(0, maxSingleSide(root.left));
        final int maxRight = Math.max(0, maxSingleSide(root.right));
        //不断更新路径和
        maxPathSum = Math.max(maxPathSum, root.val + leftMax + maxRight);
        return root.val + Math.max(leftMax, maxRight);

    }

    //给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
    //
    //完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
    public int numSquares(int n) {
        //f[i] 表示最少需要多少个数的平方来表示整数i
        //f[0]=0
        //f[i]=1+ min(f[i-j^2),j^2<=j
        int[] f = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int minn = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                minn = Math.min(minn, f[i - j * j]);
            }
            f[i] = minn + 1;
        }
        return f[n];
    }

    //给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
    //数组中的每个元素代表你在该位置可以跳跃的最大长度。
    //判断你是否能够到达最后一个下标。
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int index = 0;//当前能跳到的最远的距离(坐标)
        final int lastIndex = nums.length - 1;
        for (int i = 0; i <= index; i++) {
            //不断更新距离
            index = Math.max(index, i + nums[i]);
            if (index >= lastIndex) {
                //可以跳到最后一个坐标
                return true;
            }
        }
        return false;
    }

    //给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
    //数组中的每个元素代表你在该位置可以跳跃的最大长度。
    //返回可以跳到最后一个最后一个坐标的最小跳跃数
    public int jump(int[] nums) {
        int[] minSteps = new int[nums.length];
        for (int i = 1; i < nums.length; i++) {
            minSteps[i] = Integer.MAX_VALUE;
        }
        final int lastIndex = nums.length - 1;
        for (int i = 1; i <= lastIndex; i++) {
            for (int j = 0; j < i; j++) {
                if (j + nums[j] >= i) {
                    minSteps[i] = Math.min(minSteps[j] + 1, minSteps[i]);
                }
            }
        }
        return minSteps[lastIndex];
    }

    public int jumpBest(int[] nums) {
        int steps = 0;
        int maxPos = 0;
        int curStepEnd = 0;
        //不访问最后的节点，nums[nums.length-1]一定用不上
        for (int i = 0; i < nums.length - 1; i++) {
            //当前step最远可以到的距离
            maxPos = Math.max(maxPos, i + nums[i]);
            if (i == curStepEnd) {
                //更新当前step可以走到的最远距离
                curStepEnd = maxPos;
                //进入下次跳跃
                steps++;
            }
        }
        return steps;
    }

    //给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
    //请你返回所有和为 0 且不重复的三元组。
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 3) {
            return res;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) {
                //后面都更大了，和不会为0了
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int twoSum = -nums[i];
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                final int delta = nums[left] + nums[right] - twoSum;
                if (delta == 0) {
                    res.add(Arrays.asList(nums[i], nums[left++], nums[right--]));
                    //去重，不能和前一个数一样
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right + 1]) {
                        right--;
                    }
                } else if (delta > 0) {
                    right--;
                } else {
                    left++;
                }
            }
        }
        return res;

    }

    //给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
    //返回这三个数的和。
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        int minAbs = Math.abs(res - target);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int targetTwoSum = target - nums[i];
            if (targetTwoSum <= 0 && nums[i + 1] > 0) {
                break;
            }
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                final int sum = nums[i] + nums[left] + nums[right];
                if (sum - target == 0) {
                    return target;
                }
                if (Math.abs(sum - target) < Math.abs(res - target)) {
                    res = sum;
                }
                if (sum > target) {
                    //如果right不变，left增大，那么sum只会增大，更大于target了，所以此时应当减小right
                    right--;
                    //去重
                    while (left < right && nums[right] == nums[right + 1]) {
                        right--;
                    }
                } else {
                    //如果left不变，right减小，sum只会减小，更小于target了，所以此时应当增大left
                    left++;
                    //去重
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                }
            }
        }
        return res;
    }

    public static void main(String[] args) {
        final Solution1 solution = new Solution1();
//        solution.numDifferentIntegers("0a0");
//        solution.kthSmallest(stringToMatrix("[[1,3,5],[6,7,12],[11,14,14]]"), 3);
//        solution.topKFrequent(stringToArray("[1,1,1,2,2,3]"), 2);
//        solution.maxSlidingWindow(stringToArray("[1,-1]"), 1);
//        final Codec codec = new Codec();
//        final TreeNode node = codec.deserialize("(1,2,3,X,X,4,5)");
//        Codec ser = new Codec();
//        Codec deser = new Codec();
//        TreeNode ans = deser.deserialize(ser.serialize(node));
//        solution.longestConsecutive(stringToArray("[100,4,200,1,3,2]"));
//        ListNode head = new ListNode(1);
//        head.next = new ListNode(2);
//        solution.reverseList(head);
//        String s = solution.largestNumber(new int[]{3, 30, 34, 5, 9});
//        solution.jump(new int[]{2, 3, 1, 1, 4});
        solution.threeSumClosest(new int[]{4, 0, 5, -5, 3, 3, 0, -4, -5}, -2);
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
