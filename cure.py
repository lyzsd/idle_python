import matplotlib.pyplot as plt
import numpy as np
little_freq = [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600,
               1497600, 1612800, 1708800, 1804800]

big_freq = [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800,
            2112000, 2227200, 2342400, 2419200]

s_big_freq = [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400,
              2265600, 2380800, 2496000, 2592000, 2688000, 2764800, 2841600]

small_dyn = [2, 3, 4, 5, 5, 7, 8, 10, 11, 13, 15, 17, 20, 23, 26, 29]
for small_d in range(16):
    small_dyn[small_d] = small_dyn[small_d] * 2 * 4
big_dyn = [30, 32, 34, 37, 38, 41, 43, 47, 50, 52, 56, 60, 64, 68, 73, 78]
for small_d in range(16):
    big_dyn[small_d] = big_dyn[small_d] * 8 * 3



small_static = [42, 42, 42, 42, 42, 44, 46, 49, 51, 55, 58, 61, 66, 69, 72, 75]
big_static = [127, 135, 141, 148, 152, 160, 166, 175, 182, 188, 197, 206, 216, 225, 235, 245]
small_power = []
small_cost = []
big_power = []
big_no_static = []
big_cost =[]
for small_util in range(325):
    raw_freq = int(small_util * 1.25 * 1804800 / 325)
    util_to_freq = little_freq[np.searchsorted(little_freq, raw_freq)] if little_freq[0] <= raw_freq <= little_freq[-1] \
        else (little_freq[0] if raw_freq < little_freq[0] else little_freq[-1])
    cur_cap = int(util_to_freq / little_freq[-1] * 325)
    small_dyn_power = small_dyn[little_freq.index(util_to_freq)] * small_util / cur_cap
    small_static_power = small_static[little_freq.index(util_to_freq)]
    small_power.append((small_dyn_power + small_static_power))
    small_cost.append((small_dyn_power + small_static_power) / cur_cap)

for big_util in range(828):
    raw_freq = int(big_util * 1.25 * 2419200 / 828)
    util_to_freq = big_freq[np.searchsorted(big_freq, raw_freq)] if big_freq[0] <= raw_freq <= big_freq[-1] else (
        big_freq[0] if raw_freq < big_freq[0] else big_freq[-1])
    cur_cap = int(util_to_freq / big_freq[-1] * 828)
    dyn_power = big_dyn[big_freq.index(util_to_freq)] * big_util  / 828
    static_power = big_static[big_freq.index(util_to_freq)]
    big_power.append((dyn_power + static_power))
    big_no_static.append(dyn_power)
    big_cost.append((dyn_power + static_power) / cur_cap)
# plt.plot(range(325), small_power)
print(big_power)
plt.plot(range(828), big_no_static)
plt.plot(range(828), big_power)
plt.show()
