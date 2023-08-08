@echo off & setlocal EnableDelayedExpansion
adb shell input tap 487 177
ping -n 3 127.0.0.1 > nul
adb shell input tap 94 578
ping -n 5 127.0.0.1 > nul
for /l %%i in (1,1,30) do (
@REM 滑动
adb shell input swipe 336 1729 190 757 300
ping -n 3 127.0.0.1 > nul)
ping -n 3 127.0.0.1 > nul
adb shell input tap 308 2314
ping -n 2 127.0.0.1 > nul
adb shell input tap 308 2314
ping -n 2 127.0.0.1 > nul
adb shell input tap 329 2178
ping -n 2 127.0.0.1 > nul
adb shell input swipe 336 1729 190 757 500
ping -n 2 127.0.0.1 > nul
adb shell input swipe 336 1729 190 757 500
ping -n 2 127.0.0.1 > nul


