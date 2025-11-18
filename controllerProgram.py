import os
import shutil

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

        # Destination folder two directories down
        dest_folder = os.path.join(os.getcwd(), "src", "east")
        os.makedirs(dest_folder, exist_ok=True)  # create if it doesn't exist

        # Destination paths
        news_dest = os.path.join(dest_folder, "feedcsv.csv")
        price_dest = os.path.join(dest_folder, "feedcsv2.csv")

        # Copy files to destination
        shutil.copy(news_src, news_dest)
        shutil.copy(price_src, price_dest)

        print(f"Copied files successfully to: {dest_folder}")
        break

    else:
        print("demo does not exist — try again.")
