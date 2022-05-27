from typing import List

import bisect

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if len(nums) == 0:return
        l = []
        ans = 0
        for i in nums:
            index = bisect.bisect_right(l, i)
            l[index:index] = [i]
            ans += len(l) - index - 1
        return ans

test = Solution()
nums =  [7,5,6,4,4]
print(test.reversePairs(nums))