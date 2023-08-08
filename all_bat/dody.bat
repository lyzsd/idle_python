@echo off & setlocal EnableDelayedExpansion
adb shell input tap 982 2157
ping -n 5 127.0.0.1 > nul
adb shell input tap 460 797
ping -n 5 127.0.0.1 > nul
adb shell input tap 453 885
ping -n 5 127.0.0.1 > nul
adb shell input tap 192 1836
ping -n 5 127.0.0.1 > nul
set /a count=15
for /l %%i in (1,1,30) do (
@REM 滑动
adb shell input swipe 336 1729 190 757 500
ping -n 3 127.0.0.1 > nul)
ping -n 30 127.0.0.1 > nul
adb shell input tap 303 2309
ping -n 2 127.0.0.1 > nul
adb shell input tap 303 2309
ping -n 2 127.0.0.1 > nul
adb shell input tap 303 2309
ping -n 2 127.0.0.1 > nul



