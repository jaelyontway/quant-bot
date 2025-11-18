import os
import shutil
import time
import subprocess

while True:
    date = input('Enter a date (YYYY-MM-DD): ').strip()

    # Folder and filenames
    folder_name = f"NVDA_{date}"
    news_file = f"news_{date}.csv"
    price_file = f"nvda_prices_{date}.csv"
    
    # Full paths to source files
    news_src = os.path.join("data", "demo_data", folder_name, news_file)
    price_src = os.path.join("data", "demo_data", folder_name, price_file)

    # Check if both exist
    if os.path.isfile(news_src) and os.path.isfile(price_src):
        print("success")

        # Destination paths in current working directory
        news_dest = os.path.join(os.getcwd(), "feedcsv.csv")
        price_dest = os.path.join(os.getcwd(), "feedcsv2.csv")

        # Copy files to current directory
        shutil.copy(news_src, news_dest)
        shutil.copy(price_src, price_dest)
        print(f"Copied files successfully to current directory.")

        # Pause for 0.5 seconds
        time.sleep(0.5)

        # Run the east model script in src/east
        dest_folder = os.path.join(os.getcwd(), "src", "east")
        run_script = os.path.join(dest_folder, "run_east_model.py")
        if os.path.isfile(run_script):
            subprocess.run(["python3", run_script], cwd=dest_folder)
        else:
            print(f"run_east_model.py not found in {dest_folder}")

        break

    else:
        print("demo does not exist — try again.")
