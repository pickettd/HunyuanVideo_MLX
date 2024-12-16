import psutil
import time
import os
import curses
from datetime import datetime
import subprocess
import mlx.core as mx

def get_metal_memory():
    """Get Metal GPU memory usage using system_profiler"""
    try:
        cmd = ['system_profiler', 'SPDisplaysDataType']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse the output to find Metal GPU memory
        lines = result.stdout.split('\n')
        total_mem = 0
        for line in lines:
            if 'VRAM' in line:
                # Extract memory value in MB/GB and convert to GB
                mem_str = line.split(':')[1].strip()
                if 'MB' in mem_str:
                    total_mem = float(mem_str.replace('MB', '').strip()) / 1024
                elif 'GB' in mem_str:
                    total_mem = float(mem_str.replace('GB', '').strip())
        
        # Get current memory usage through psutil as an approximation
        memory = psutil.virtual_memory()
        used_mem = (memory.total - memory.available) / (1024 ** 3)
        
        return {
            'total': total_mem,
            'used': min(used_mem, total_mem),  # Cap at total memory
            'free': max(0, total_mem - used_mem)
        }
    except Exception as e:
        return None

def get_mlx_info():
    """Get MLX-specific information"""
    try:
        return {
            'version': mx.__version__,
            'backend': 'Metal',
            'device': mx.default_device()
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
    
    # Get MLX info once
    mlx_info = get_mlx_info()
    
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
            
            # Get Metal memory
            metal_mem = get_metal_memory()
            
            # Display header
            stdscr.addstr(0, 0, f"HunyuanVideo MLX Resource Monitor - {current_time}", curses.color_pair(1) | curses.A_BOLD)
            
            # Display MLX info
            if mlx_info:
                stdscr.addstr(2, 0, "MLX Configuration:", curses.A_BOLD)
                stdscr.addstr(3, 2, f"Version: {mlx_info['version']}")
                stdscr.addstr(4, 2, f"Backend: {mlx_info['backend']}")
                stdscr.addstr(5, 2, f"Device: {mlx_info['device']}")
            
            # Display system resources
            stdscr.addstr(7, 0, "System Resources:", curses.A_BOLD)
            
            # Display CPU usage
            color = curses.color_pair(1) if cpu_percent < 70 else curses.color_pair(2) if cpu_percent < 90 else curses.color_pair(3)
            stdscr.addstr(8, 2, f"CPU Usage: {cpu_percent:>5.1f}%", color)
            
            # Display memory usage
            color = curses.color_pair(1) if memory_percent < 70 else curses.color_pair(2) if memory_percent < 90 else curses.color_pair(3)
            stdscr.addstr(9, 2, f"Memory: {memory.used/1024/1024/1024:>5.1f}GB / {memory.total/1024/1024/1024:>5.1f}GB ({memory_percent:>5.1f}%)", color)
            
            # Display Metal memory if available
            if metal_mem:
                metal_percent = (metal_mem['used'] / metal_mem['total']) * 100 if metal_mem['total'] > 0 else 0
                color = curses.color_pair(1) if metal_percent < 70 else curses.color_pair(2) if metal_percent < 90 else curses.color_pair(3)
                stdscr.addstr(10, 2, f"Metal Memory: {metal_mem['used']:>5.1f}GB / {metal_mem['total']:>5.1f}GB ({metal_percent:>5.1f}%)", color)
            
            # Display Python processes
            stdscr.addstr(12, 0, "Python Processes:", curses.A_BOLD)
            row = 13
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
