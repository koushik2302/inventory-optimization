@echo off
title Inventory Optimizer
echo ========================================================
echo Starting Inventory Optimization App...
echo ========================================================
echo.

:: Activate the conda environment
call C:\Users\chunc\miniconda3\Scripts\activate.bat C:\Users\chunc\miniconda3\envs\inv-opt

:: Start ollama in the background (just in case)
start /b ollama serve >nul 2>&1

:: Wait a couple seconds
timeout /t 2 /nobreak >nul

:: Launch Streamlit app
echo Starting Streamlit interface...
python -m streamlit run agent/app.py
