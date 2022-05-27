class Solution:
    def findNthDigit(self, n: int) -> int:
        if n == 0:return 0
        num = 0
        pre = 0
        digit = 1
        while n > num:
            pre = num
            num += digit * (9 * 10 ** (digit - 1))
            digit += 1
        res = n - pre - 1
        digit -= 1
        count = res // (digit)
        number  = 10 ** (digit - 1) + count
        rest = res % digit
        return int(str(number)[rest])

test = Solution()
n = 3
print(test.findNthDigit(n))