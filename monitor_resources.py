import psutil
import time
import os
import curses
from datetime import datetime

def get_gpu_memory():
    """Get MPS GPU memory usage"""
    try:
        # This is a basic approximation since MPS doesn't provide direct memory stats
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 ** 3),  # Convert to GB
            'used': (memory.total - memory.available) / (1024 ** 3),
            'free': memory.available / (1024 ** 3)
        }
    except Exception as e:
        return None

def get_process_memory(pid):
    """Get memory usage for a specific process"""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 ** 3)  # Convert to GB
    except:
        return 0

def main(stdscr):
    # Set up colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    
    # Hide cursor
    curses.curs_set(0)
    
    # Get initial process list
    python_processes = {p.pid: p.cmdline() for p in psutil.process_iter(['pid', 'cmdline']) 
                       if 'python' in ' '.join(p.cmdline()).lower()}

    while True:
        try:
            stdscr.clear()
            
            # Get current time
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Get system memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get GPU memory
            gpu_mem = get_gpu_memory()
            
            # Display header
            stdscr.addstr(0, 0, f"HunyuanVideo Resource Monitor - {current_time}", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(2, 0, "System Resources:", curses.A_BOLD)
            
            # Display CPU usage
            color = curses.color_pair(1) if cpu_percent < 70 else curses.color_pair(2) if cpu_percent < 90 else curses.color_pair(3)
            stdscr.addstr(3, 2, f"CPU Usage: {cpu_percent:>5.1f}%", color)
            
            # Display memory usage
            color = curses.color_pair(1) if memory_percent < 70 else curses.color_pair(2) if memory_percent < 90 else curses.color_pair(3)
            stdscr.addstr(4, 2, f"Memory: {memory.used/1024/1024/1024:>5.1f}GB / {memory.total/1024/1024/1024:>5.1f}GB ({memory_percent:>5.1f}%)", color)
            
            # Display GPU memory if available
            if gpu_mem:
                gpu_percent = (gpu_mem['used'] / gpu_mem['total']) * 100
                color = curses.color_pair(1) if gpu_percent < 70 else curses.color_pair(2) if gpu_percent < 90 else curses.color_pair(3)
                stdscr.addstr(5, 2, f"GPU Memory: {gpu_mem['used']:>5.1f}GB / {gpu_mem['total']:>5.1f}GB ({gpu_percent:>5.1f}%)", color)
            
            # Display Python processes
            stdscr.addstr(7, 0, "Python Processes:", curses.A_BOLD)
            row = 8
            for pid, cmdline in python_processes.items():
                if psutil.pid_exists(pid):
                    mem_usage = get_process_memory(pid)
                    cmd_str = ' '.join(cmdline)
                    if 'sample_video_mps.py' in cmd_str:
                        color = curses.color_pair(1)
                    else:
                        color = curses.A_NORMAL
                    stdscr.addstr(row, 2, f"PID {pid:>6}: {mem_usage:>5.1f}GB - {cmd_str[:60]}", color)
                    row += 1
            
            # Display help
            stdscr.addstr(row + 2, 0, "Press 'q' to quit", curses.color_pair(2))
            
            # Refresh the screen
            stdscr.refresh()
            
            # Check for 'q' key press
            stdscr.timeout(1000)  # Wait up to 1 second for key press
            key = stdscr.getch()
            if key == ord('q'):
                break
            
        except KeyboardInterrupt:
            break
        except curses.error:
            pass

if __name__ == "__main__":
    curses.wrapper(main)
