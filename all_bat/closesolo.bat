@echo off & setlocal EnableDelayedExpansion

@REM 切出后台应用界面
adb shell input tap 766 2337
ping 192.0.2.2 -n 1 -w 2000 > nul
@REM 向上滑动 清除后台应用
adb shell input swipe 500 1400 500 500 100
ping 192.0.2.2 -n 1 -w 2000 > nul
@REM 点击HOME键 3 回到桌面
adb shell input keyevent 3
ping 192.0.2.2 -n 1 -w 2000 > nul