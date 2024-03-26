@echo off
call C:\Users\DanCandrea\Anaconda3\condabin\activate.bat
call activate VidRecog2
python hand_trajectory_extraction.py
python click_detection.py
call conda deactivate
pause