@echo off & setlocal EnableDelayedExpansion


@REM 点击提前缓存的第一条视频
adb shell input tap 200 900
ping -n 4 127.0.0.1 > nul

@REM 播放历史第一条
adb shell input tap 200 900
ping -n 150 127.0.0.1 > nul



@REM 点击HOME键 3 回到桌面
adb shell input keyevent 3
ping -n 2 127.0.0.1 > nul